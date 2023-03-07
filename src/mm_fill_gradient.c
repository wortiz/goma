#include "mm_fill_gradient.h"

#include "el_elm.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "mm_fill_stress_legacy.h"
#include "rf_fem.h"
#include "rf_fem_const.h"

int assemble_gradient(dbl tt, /* parameter to vary time integration from
                               * explicit (tt = 1) to implicit (tt = 0) */
                      dbl dt) /* current time step size */
{
  int dim;
  int p, q, a, b;

  int eqn, var;
  int peqn, pvar;
  int i, j;
  int status;

  dbl h3;          /* Volume element (scale factors). */
  dbl dh3dmesh_pj; /* Sensitivity to (p,j) mesh dof. */

  dbl grad_v[DIM][DIM];
  dbl g[DIM][DIM]; /* velocity gradient tensor */

  dbl det_J; /* determinant of element Jacobian */

  dbl d_det_J_dmesh_pj; /* for specific (p,j) mesh dof */

  dbl advection;
  dbl advection_a, advection_b;
  dbl source;

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

  int R_g[DIM][DIM];
  int v_g[DIM][DIM];

  status = 0;

  /*
   * Unpack variables from structures for local convenience...
   */

  dim = pd->Num_Dim;

  eqn = R_GRADIENT11;

  /*
   * Bail out fast if there's nothing to do...
   */

  if (!pd->e[pg->imtrx][eqn]) {
    return (status);
  }

  wt = fv->wt;

  det_J = bf[eqn]->detJ; /* Really, ought to be mesh eqn. */

  h3 = fv->h3; /* Differential volume element (scales). */

  /* load eqn and variable number in tensor form */

  v_g[0][0] = VELOCITY_GRADIENT11;
  v_g[0][1] = VELOCITY_GRADIENT12;
  v_g[1][0] = VELOCITY_GRADIENT21;
  v_g[1][1] = VELOCITY_GRADIENT22;
  v_g[0][2] = VELOCITY_GRADIENT13;
  v_g[1][2] = VELOCITY_GRADIENT23;
  v_g[2][0] = VELOCITY_GRADIENT31;
  v_g[2][1] = VELOCITY_GRADIENT32;
  v_g[2][2] = VELOCITY_GRADIENT33;

  R_g[0][0] = R_GRADIENT11;
  R_g[0][1] = R_GRADIENT12;
  R_g[1][0] = R_GRADIENT21;
  R_g[1][1] = R_GRADIENT22;
  R_g[0][2] = R_GRADIENT13;
  R_g[1][2] = R_GRADIENT23;
  R_g[2][0] = R_GRADIENT31;
  R_g[2][1] = R_GRADIENT32;
  R_g[2][2] = R_GRADIENT33;

  /*
   * Field variables...
   */

  /*
   * In Cartesian coordinates, this velocity gradient tensor will
   * have components that are...
   *
   * 			grad_v[a][b] = d v_b
   *				       -----
   *				       d x_a
   */

  for (a = 0; a < VIM; a++) {
    for (b = 0; b < VIM; b++) {
      grad_v[a][b] = fv->grad_v[a][b];
    }
  }

  for (a = 0; a < VIM; a++) {
    for (b = 0; b < VIM; b++) {
      g[a][b] = fv->G[a][b];
    }
  }

  /*
   * Residuals_________________________________________________________________
   */

  if (af->Assemble_Residual) {
    /*
     * Assemble each component "ab" of the velocity gradient equation...
     */
    for (a = 0; a < VIM; a++) {
      for (b = 0; b < VIM; b++) {
        eqn = R_g[a][b];
        /*
         * In the element, there will be contributions to this many equations
         * based on the number of degrees of freedom...
         */

        for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {

          wt_func = bf[eqn]->phi[i]; /* add Petrov-Galerkin terms as necessary */

          advection = 0.;

          if (upd->devss_traceless_gradient) {
            if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
              advection -= grad_v[a][b] - fv->div_v * delta(a, b) / ((dbl)VIM);
              advection *= wt_func * det_J * wt * h3;
              advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
            }
          } else {
            if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
              advection -= grad_v[a][b];
              advection *= wt_func * det_J * wt * h3;
              advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
            }
          }

          /*
           * Source term...
           */

          source = 0;

          if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
            source += g[a][b];
            source *= wt_func * det_J * h3 * wt;
            source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
          }

          lec->R[LEC_R_INDEX(upd->ep[pg->imtrx][eqn], i)] += advection + source;
        }
      }
    }
  }

  /*
   * Jacobian terms...
   */

  if (af->Assemble_Jacobian) {
    for (a = 0; a < VIM; a++) {
      for (b = 0; b < VIM; b++) {
        eqn = R_g[a][b];
        peqn = upd->ep[pg->imtrx][eqn];

        for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
          wt_func = bf[eqn]->phi[i]; /* add Petrov-Galerkin terms as necessary */

          /*
           * J_G_v
           */
          for (p = 0; p < WIM; p++) {
            var = VELOCITY1 + p;
            if (pd->v[pg->imtrx][var]) {
              pvar = upd->vp[pg->imtrx][var];
              for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                phi_j = bf[var]->phi[j];

                advection = 0.;

                if (upd->devss_traceless_gradient) {
                  dbl div_phi_j_e_p = 0.;
                  for (int b = 0; b < VIM; b++) {
                    div_phi_j_e_p += bf[var]->grad_phi_e[j][p][b][b];
                  }
                  if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                    advection -=
                        bf[var]->grad_phi_e[j][p][a][b] - div_phi_j_e_p * delta(a, b) / ((dbl)VIM);
                    advection *= wt_func * det_J * wt * h3;
                    advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
                  }
                } else {
                  if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                    advection -= bf[var]->grad_phi_e[j][p][a][b];
                    advection *= wt_func * det_J * wt * h3;
                    advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
                  }
                }

                source = 0.;

                lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += advection + source;
              }
            }
          }

          /*
           * J_G_d
           */
          for (p = 0; p < dim; p++) {
            var = MESH_DISPLACEMENT1 + p;
            if (pd->v[pg->imtrx][var]) {
              pvar = upd->vp[pg->imtrx][var];
              for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                phi_j = bf[var]->phi[j];

                d_det_J_dmesh_pj = bf[eqn]->d_det_J_dm[p][j];

                dh3dmesh_pj = fv->dh3dq[p] * bf[var]->phi[j];

                advection = 0.;

                if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                  /*
                   * three parts:
                   *    advection_a =
                   *    	Int ( d(Vv)/dmesh h3 |Jv| )
                   *
                   *    advection_b =
                   *  (i)	Int ( Vv h3 d(|Jv|)/dmesh )
                   *  (ii)      Int ( Vv dh3/dmesh |Jv|   )
                   */

                  advection_a = -grad_v[a][b];

                  advection_a *= (d_det_J_dmesh_pj * h3 + det_J * dh3dmesh_pj);

                  advection_b = -fv->d_grad_v_dmesh[a][b][p][j];

                  advection_b *= det_J * h3;

                  advection = advection_a + advection_b;

                  advection *= wt_func * wt * pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
                }

                source = 0.;

                if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                  source += g[a][b];

                  source *= d_det_J_dmesh_pj * h3 + det_J * dh3dmesh_pj;

                  source *= wt_func * wt * pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
                }

                lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += advection + source;
              }
            }
          }

          /*
           * J_G_G
           */

          for (p = 0; p < VIM; p++) {
            for (q = 0; q < VIM; q++) {
              var = v_g[p][q];

              if (pd->v[pg->imtrx][var]) {
                pvar = upd->vp[pg->imtrx][var];
                for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                  phi_j = bf[var]->phi[j];

                  source = 0.;

                  if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                    if ((a == p) && (b == q)) {
                      source = phi_j * det_J * h3 * wt_func * wt *
                               pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
                    }
                  }

                  lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += source;
                }
              }
            }
          }
        }
      }
    }
  }
  return (status);
}