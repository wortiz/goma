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

/* mm_fill_rs -- Routine for  matrix & residual assembly for real ALE solid */

/* Standard include files */

#include <math.h>
#include <stdio.h>
#include <string.h>

/* GOMA include files */

#include "mm_fill_rs.h"

#include "density.h"
#include "el_elm.h"
#include "exo_struct.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "mm_eh.h"
#include "mm_elem_block_structs.h"
#include "mm_fill_solid.h"
#include "mm_fill_terms.h"
#include "mm_mp.h"
#include "mm_mp_const.h"
#include "mm_mp_structs.h"
#include "mm_qtensor_model.h"
#include "mm_shell_util.h"
#include "rd_mesh.h"
#include "rf_allo.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_io.h"
#include "rf_mp.h"
#include "std.h"

#define GOMA_MM_FILL_RS_C

/*********** R O U T I N E S   I N   T H I S   F I L E ***********************
 *
 *						-All routines in this
 *				NOTE:		 file are called only by
 *						 rf_fill.c: matrix_fill
 *						 except possibly for static
 *						 functions.
 *
 *       NAME			TYPE			CALLED BY
 *  -----------------------------------------------------------------
 *  assemble_real_solid		void	                matrix_fill
 *
 *
 ******************************************************************************/

/*  _______________________________________________________________________  */

/* assemble_real_solid -- assemble terms (Residual &| Jacobian) for solid stress eqns
 *
 * in:
 * 	ei -- pointer to Element Indeces		structure
 *	pd -- pointer to Problem Description		structure
 *	af -- pointer to Action Flag			structure
 *	bf -- pointer to Basis Function			structure
 *	fv -- pointer to Field Variable			structure
 *	cr -- pointer to Constitutive Relation		structure
 *	mp -- pointer to Material Property		structure
 *	esp-- pointer to Element Stiffness Pointers	structure
 *	lec-- pointer to Local Element Contribution	structure
 *
 * out:
 *	a   -- gets loaded up with proper contribution
 *	lec -- gets loaded up with local contributions to resid, Jacobian
 * 	r   -- residual RHS vector
 *
 * Created:	Thu Mar  3 07:48:01 MST 1994 pasacki@sandia.gov
 *
 * Revised:	Sat Mar 19 16:07:51 MST 1994 pasacki@sandia.gov
 *
 *
 * Note: currently we do a "double load" into the addresses in the global
 *       "a" matrix and resid vector held in "esp", as well as into the
 *       local accumulators in "lec".
 *
 */

int assemble_real_solid(double time_value, double tt, double dt) {
  /*
   * Some local variables for convenience...
   */

  int eqn, peqn;
  int var, pvar;
  int dim;
  int p, q, a, b, c;

  int w;

  int i, j;
  int status, err;

  dbl det_J;
  dbl d_det_J_dmeshbj;
  dbl h3;          /* Volume element (scale factors). */
  dbl dh3dmesh_bj; /* Sensitivity to (b,j) mesh dof. */

  dbl VV[DIM][DIM];           /* Inertia Tensor... */
  double vconv[MAX_PDIM];     /*Calculated convection velocity */
  double vconv_old[MAX_PDIM]; /*Calculated convection velocity at previous time*/
  CONVECTION_VELOCITY_DEPENDENCE_STRUCT d_vconv_struct;
  CONVECTION_VELOCITY_DEPENDENCE_STRUCT *d_vconv = &d_vconv_struct;

  dbl TT[DIM][DIM];                    /* Real solid stress tensor... */
  dbl dTT_dx[DIM][DIM][DIM][MDE];      /* Sensitivity of stress tensor...
                          to nodal displacements */
  dbl dTT_drs[DIM][DIM][DIM][MDE];     /* Sensitivity of stress tensor...
                          to real solid  displacements */
  dbl dTT_dp[DIM][DIM][MDE];           /* Sensitivity of stress tensor...
                                  to nodal pressure*/
  dbl dTT_dc[DIM][DIM][MAX_CONC][MDE]; /* Sensitivity of stress tensor...
                          to nodal concentration*/
  dbl dTT_dp_liq[DIM][DIM][MDE];       /* Sensitivity of stress tensor...
                                           to nodal porous liquid pressure*/
  dbl dTT_dp_gas[DIM][DIM][MDE];       /* Sensitivity of stress tensor...
                                           to nodal porous gas pressure*/
  dbl dTT_dporosity[DIM][DIM][MDE];    /* Sensitivity of stress tensor...
                                        to nodal porosity*/
  dbl dTT_dT[DIM][DIM][MDE];           /* Sensitivity of stress tensor...
                                      to nodal temperature*/
  dbl dTT_dmax_strain[DIM][DIM][MDE];  /* Sensitivity of stress tensor...
                                          to nodal max_strain*/

  dbl mu;
  dbl lambda, rho;

  dbl g[DIM]; /* solid body force. */

  dbl diff_a = 0, diff_b = 0, diff_c = 0; /* Temporary variables hold partially */
  /* constructed diffusion terms... */
  dbl mass, diffusion, advection = 0, advect_a, advect_b, advect_c;

  dbl source;

  dbl wt;

  /*
   * Galerkin weighting functions for i-th and a-th mesh stress residuals
   * and some of their derivatives...
   */

  dbl phi_i;
  dbl grad_phi_i_e_a[DIM][DIM];

  /*
   * Interpolation functions for variables and some of their derivatives.
   */

  dbl phi_j;

  /*
   * Mesh derivatives...
   */
  dbl dgradphi_i_e_a_dmeshbj[DIM][DIM]; /* for specific (i, a, b, j) !!  */

  /*finally we have some time-derivative quantities for the Newmark scheme */
  dbl x_c[DIM];           /*base coordinates of deformed mesh */
  dbl x_old[DIM];         /*old value of base coordinates on deformed mesh */
  dbl x_dot_old[DIM];     /*Old value of mesh velocity */
  dbl x_dbl_dot_old[DIM]; /*old value of acceleration */
  dbl x_dbl_dot[DIM];     /*current value of mesh acceleration */
  dbl newmark_beta;       /*Newmark scheme beta value. */

  status = 0;

  /*
   * Unpack variables from structures for local convenience...
   */

  dim = goma_ctx->fem.pd->Num_Dim;

  eqn = R_SOLID1; /* Well, yes, there really are 3, */
                  /* but for now just use the first */
                  /* to tell whether there is anything */
                  /* to do...*/

  wt = goma_ctx->fem.fv->wt;
  h3 = goma_ctx->fem.fv->h3; /* Differential volume element (scales). */

  /*
   * Bail out fast if there's nothing to do...
   */

  if (!goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn]) {
    return (status);
  }

  det_J = goma_ctx->fem.bf[eqn]->detJ;

  /*
   * Material property constants, etc. Any variations for this
   * Gauss point were evaluated in mm_fill.
   */
  rho = density(NULL, time_value);

  mu = goma_ctx->fem.elc_rs->lame_mu;

  lambda = goma_ctx->fem.elc_rs->lame_lambda;

  if (goma_ctx->fem.mp->RealSolidSourceModel == CONSTANT) {
    for (a = 0; a < dim; a++) {
      g[a] = goma_ctx->fem.mp->real_solid_source[a];
    }
  } else {
    GOMA_EH(GOMA_ERROR, "Unrecognized RealSolidSourceModel");
  }

  /*
   * Get the deformation gradients and tensors for the real solid case
   */

  err = belly_flop_rs(mu);
  GOMA_EH(err, "error in belly flop");
  if (err == 2)
    return (err);

  /*
   * Total mesh stress tensor...
   */
  /* initialize some arrays */
  memset(TT, 0, sizeof(double) * DIM * DIM);
  if (goma_ctx->fem.af->Assemble_Jacobian) {
    memset(dTT_dx, 0, sizeof(double) * DIM * DIM * DIM * MDE);
    memset(dTT_drs, 0, sizeof(double) * DIM * DIM * DIM * MDE);
    memset(dTT_dp, 0, sizeof(double) * DIM * DIM * MDE);
    memset(dTT_dc, 0, sizeof(double) * DIM * DIM * MAX_CONC * MDE);
    memset(dTT_dp_liq, 0, sizeof(double) * DIM * DIM * MDE);
    memset(dTT_dp_gas, 0, sizeof(double) * DIM * DIM * MDE);
    memset(dTT_dporosity, 0, sizeof(double) * DIM * DIM * MDE);
    memset(dTT_dT, 0, sizeof(double) * DIM * DIM * MDE);
    memset(dTT_dmax_strain, 0, sizeof(double) * DIM * DIM * MDE);
  }

  /* NB: Note I (prs) have created a separate stress tensor routine than
     the one used for the mesh stress (or pseudo stress), because of
    the different strains and potentially different properties.  This
    is basically a matter of where the "if" block goes.  However, I have recently
    upgraded this routine (3/15/2012) to use the exact same "load_elastic_properties"
    so that any time a new lame coeff. model is added, it is available for
    both TOTAL_ALE and the traditional LAGRANGIAN
  */

  err = solid_stress_tensor(TT, dTT_dx, dTT_drs, dTT_dp, dTT_dc, dTT_dp_liq, dTT_dp_gas,
                            dTT_dporosity, dTT_dT, dTT_dmax_strain, mu, lambda);

  /* Calculate inertia of mesh if required */
  if (goma_ctx->fem.cr->MeshMotion == LAGRANGIAN ||
      goma_ctx->fem.cr->MeshMotion == DYNAMIC_LAGRANGIAN ||
      goma_ctx->fem.cr->MeshMotion == TOTAL_ALE) {
    /* NB: DITTO on above comment*/
    /* Actually, we need another one of these for an arbitrary
      advected lagrangian velocity */

    /* As of 12/17/98 this call could be replaced by a call to the
       get_convection_velocity counterpart for the pseudo-solid. However
       we need to make sure that the deformation gradient that we want
       is the pseudo-solids here.  this routine has been changed to use
       that.  When you eliminate it in favor of a call to its counterpart
       just put the TOTAL_ALE logic in the counterpart
       get_convection_velocity */

    /* As of 6/14/99 we are changing this back to the original case.  I.E.
       we will use the Material deformation gradient and not the mesh
       deformation gradient for this velocity component. Reasoning is that
       arbitrary kinematics govern the mesh, and can be independent motions
       of the material.   This goes into no-slip and advective inertia, and
       hence should be that of the material.  */

    err = get_convection_velocity_rs(vconv, vconv_old, d_vconv, dt, tt);

    /* Now add on the advection due to Purly Eulerian moving of the solid through
     * the mesh, as would occur in the Level-set solid eulerian case */
    if (goma_ctx->fem.pd->TimeIntegration != STEADY &&
        goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_ADVECTION)]) {
      for (a = 0; a < dim; a++) {
        vconv[a] = goma_ctx->fem.fv_dot->d_rs[a];
        for (b = 0; b < goma_ctx->problem.vim; b++) {
          vconv[a] -= goma_ctx->fem.fv_dot->d_rs[b] * goma_ctx->fem.fv->grad_d_rs[b][a];
        }
      }
      for (a = 0; a < goma_ctx->problem.vim; a++) {
        for (b = 0; b < goma_ctx->problem.vim; b++) {
          var = SOLID_DISPLACEMENT1 + b;
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            phi_j = goma_ctx->fem.bf[var]->phi[j];

            d_vconv->rs[a][b][j] += phi_j * (1. + 2. * tt) * dt * delta(a, b);

            d_vconv->rs[a][b][j] -= (1. + 2. * tt) * phi_j * dt * goma_ctx->fem.fv->grad_d_rs[b][a];

            for (c = 0; c < goma_ctx->problem.vim; c++) {
              /*grad_phi_e[i][j][k][l] = e_k e_l : grad(phi_i e_j )*/
              d_vconv->rs[a][b][j] -=
                  goma_ctx->fem.fv_dot->d_rs[c] * goma_ctx->fem.bf[var]->grad_phi_e[j][b][c][a];
            }
          }
        }
      }
    }
  } else /* No inertia in an Arbitrary Mesh */
  {
    for (a = 0; a < dim; a++) {
      vconv[a] = 0.;
    }
  }
  for (a = 0; a < dim; a++) {
    for (b = 0; b < dim; b++) {
      VV[a][b] = vconv[a] * vconv[b];
    }
  }

  newmark_beta = goma_ctx->fem.tran->newmark_beta;
  if (goma_ctx->fem.pd->TimeIntegration != STEADY &&
      goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_MASS)]) {
    for (a = 0; a < dim; a++) {
      x_c[a] = goma_ctx->fem.fv->d_rs[a];
      x_old[a] = goma_ctx->fem.fv_old->d_rs[a];
      x_dot_old[a] = goma_ctx->fem.fv_dot_old->d_rs[a];
      x_dbl_dot_old[a] = goma_ctx->fem.fv_dot_dot_old->d_rs[a];

      x_dbl_dot[a] = (x_c[a] - x_old[a]) / newmark_beta / dt / dt -
                     x_dot_old[a] * (1.0 / newmark_beta / dt) -
                     x_dbl_dot_old[a] * (1 - 2. * newmark_beta) / 2. / newmark_beta;
    }
  } else {
    for (a = 0; a < dim; a++) {
      x_dbl_dot[a] = 0.;
    }
  }

  /*
   * Residuals_________________________________________________________________
   */

  if (goma_ctx->fem.af->Assemble_Residual) {
    /*
     * Assemble each component "a" of the momentum equation...
     */
    for (a = 0; a < dim; a++) {
      eqn = R_SOLID1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {
        phi_i = goma_ctx->fem.bf[eqn]->phi[i];

        mass = 0.;
        if (goma_ctx->fem.pd->TimeIntegration != STEADY &&
            goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_MASS)]) {
          if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_MASS) {
            mass = -x_dbl_dot[a];
            mass *= phi_i * rho * det_J * wt;
            mass *= h3;
            mass *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_MASS)];
          }
        }

        diffusion = 0.;
        if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_DIFFUSION) {
          for (q = 0; q < goma_ctx->problem.vim; q++) {
            for (p = 0; p < goma_ctx->problem.vim; p++) {
              grad_phi_i_e_a[p][q] = goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q];
              diffusion += grad_phi_i_e_a[p][q] * TT[q][p];
            }
          }
          diffusion *= -det_J * wt;
          diffusion *= h3;

          diffusion *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_DIFFUSION)];
        }

        /* add inertia of moving SOLID */
        advection = 0.;
        if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_ADVECTION) {
          GOMA_WH(-1, "Warning: Real_SOLID advection untested.\n");
          for (p = 0; p < dim; p++) {
            for (q = 0; q < dim; q++) {
              grad_phi_i_e_a[p][q] = goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q];

              /* PRS 4/5/2002:  This doesn't look right.  Need del(rho*v*v).  Check */

              advection += grad_phi_i_e_a[p][q] * VV[q][p];
            }
          }
          advection *= rho * det_J * wt;
          /* Advection term only applies in Lagrangian SOLID motion */
          advection *= h3;
          advection *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_ADVECTION)];
        }

        source = 0.;
        if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_SOURCE) {
          source = phi_i * g[a] * det_J * wt;
          /* Source term only applies in Lagrangian mesh motion */
          source *= h3;
          source *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_SOURCE)];
        }

        /* porous term removed for SOLID equation - the additional effects due
           to porosity are entered into the consitutive equation for stress */

        goma_ctx->fem.lec->R[LEC_R_INDEX(peqn, i)] += diffusion + source + advection;
      }
    }
  }

  /*
   * Jacobian terms...
   */

  if (goma_ctx->fem.af->Assemble_Jacobian) {
    for (a = 0; a < dim; a++) {
      eqn = R_SOLID1 + a;
      peqn = goma_ctx->fem.upd->ep[goma_ctx->fem.pg->imtrx][eqn];

      for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {
        phi_i = goma_ctx->fem.bf[eqn]->phi[i];
        /*
         * Set up some preliminaries that are needed for the (a,i)
         * equation for bunches of (b,j) column variables...
         */
        for (p = 0; p < goma_ctx->problem.vim; p++) {
          for (q = 0; q < goma_ctx->problem.vim; q++) {
            grad_phi_i_e_a[p][q] = goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q];
          }
        }

        /*
         * J_r_d
         */
        for (b = 0; b < dim; b++) {
          var = MESH_DISPLACEMENT1 + b;
          pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
              phi_j = goma_ctx->fem.bf[var]->phi[j];

              d_det_J_dmeshbj = goma_ctx->fem.bf[eqn]->d_det_J_dm[b][j];

              dh3dmesh_bj = goma_ctx->fem.fv->dh3dq[b] * phi_j;

              mass = 0.;
              if (goma_ctx->fem.pd->TimeIntegration != STEADY &&
                  goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_MASS)]) {
                mass -=
                    x_dbl_dot[a] * phi_i * rho * wt * (d_det_J_dmeshbj * h3 + dh3dmesh_bj * det_J);
                mass *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_MASS)];
              }

              diffusion = 0.;
              /* Three parts:
               *   diff_a = Int ( d(grad(phi_i e_a))/dmesh : Pi |Jv| )
               *   diff_b = Int ( grad(phi_i e_a) : d(Pi)/dmesh |Jv| )
               *   diff_c = Int ( grad(phi_i e_a) : Pi d(|Jv|)/dmesh )
               */
              if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_DIFFUSION) {

                /* LAGRANGIAN MESH - Solve in curvilinear coordinates */
                for (p = 0; p < goma_ctx->problem.vim; p++) {
                  for (q = 0; q < goma_ctx->problem.vim; q++) {
                    dgradphi_i_e_a_dmeshbj[p][q] =
                        goma_ctx->fem.bf[eqn]->d_grad_phi_e_dmesh[i][a][p][q][b][j];
                  }
                }

                diff_a = 0.;
                for (p = 0; p < goma_ctx->problem.vim; p++) {
                  for (q = 0; q < goma_ctx->problem.vim; q++) {
                    diff_a += dgradphi_i_e_a_dmeshbj[p][q] * TT[q][p];
                  }
                }
                diff_a *= -det_J * wt;
                diff_a *= h3;

                diff_b = 0.;
                for (p = 0; p < goma_ctx->problem.vim; p++) {
                  for (q = 0; q < goma_ctx->problem.vim; q++) {
                    diff_b += grad_phi_i_e_a[p][q] * dTT_dx[q][p][b][j];
                  }
                }
                diff_b *= -det_J * wt;
                diff_b *= h3;

                diff_c = 0.;
                for (q = 0; q < dim; q++) {
                  for (p = 0; p < goma_ctx->problem.vim; p++) {
                    diff_c += grad_phi_i_e_a[p][q] * TT[q][p];
                  }
                }
                diff_c *= -wt * (d_det_J_dmeshbj * h3 + det_J * dh3dmesh_bj);
              }

              diffusion = diff_a + diff_b + diff_c;

              diffusion *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_DIFFUSION)];

              /* add inertia of moving mesh */
              advect_a = 0.;
              advect_b = 0.;
              advect_c = 0.;
              if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_ADVECTION) {
                for (p = 0; p < dim; p++) {
                  for (q = 0; q < dim; q++) {
                    dgradphi_i_e_a_dmeshbj[p][q] =
                        goma_ctx->fem.bf[eqn]->d_grad_phi_e_dmesh[i][a][p][q][b][j];
                  }
                }

                for (p = 0; p < dim; p++) {
                  for (q = 0; q < dim; q++) {

                    advect_a += grad_phi_i_e_a[p][q] * VV[q][p];
                    advect_b += grad_phi_i_e_a[p][q] *
                                (d_vconv->X[q][b][j] * vconv[p] + vconv[q] * d_vconv->X[p][b][j]);
                    advect_c += dgradphi_i_e_a_dmeshbj[p][q] * VV[q][p];
                  }
                }

                advect_a *= rho * (d_det_J_dmeshbj * h3 + det_J * dh3dmesh_bj) * wt;
                advect_b *= rho * det_J * wt;
                advect_c *= rho * det_J * wt;

                advection = advect_a + advect_b + advect_c;
                advection *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_ADVECTION)];
              }

              source = 0.;

              if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_SOURCE) {
                source = phi_i * g[a] * (d_det_J_dmeshbj * h3 + det_J * dh3dmesh_bj) * wt;
                source *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][LOG2_SOURCE];
              }

              goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, i, j)] +=
                  mass + diffusion + source + advection;
            }
          }
        }

        /*
         * J_r_r
         */
        for (b = 0; b < dim; b++) {
          var = SOLID_DISPLACEMENT1 + b;
          pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
              phi_j = goma_ctx->fem.bf[var]->phi[j];

              mass = 0.;
              if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
                if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_MASS) {
                  mass = -(double)delta(a, b) * phi_j / newmark_beta / dt / dt;
                  mass *= phi_i * rho * det_J * h3 * wt;
                  mass *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_MASS)];
                }
              }

              diffusion = 0.;
              /* Only one part here because grads and Jv depend only on d:
               *   diff_b = Int ( grad(phi_i e_a) : d(Pi)/drs |Jv| )
               */
              if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_DIFFUSION) {

                diff_b = 0.;
                for (p = 0; p < goma_ctx->problem.vim; p++) {
                  for (q = 0; q < goma_ctx->problem.vim; q++) {
                    diff_b += grad_phi_i_e_a[p][q] * dTT_drs[q][p][b][j];
                  }
                }
                diff_b *= -det_J * wt;
                diff_b *= h3;
              }

              diffusion = diff_b;

              diffusion *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_DIFFUSION)];

              /* add inertia of moving mesh */
              advect_a = 0.;
              advect_b = 0.;
              advect_c = 0.;
              if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_ADVECTION) {
                for (p = 0; p < dim; p++) {
                  for (q = 0; q < dim; q++) {
                    dgradphi_i_e_a_dmeshbj[p][q] =
                        goma_ctx->fem.bf[eqn]->d_grad_phi_e_dmesh[i][a][p][q][b][j];
                  }
                }

                for (p = 0; p < dim; p++) {
                  for (q = 0; q < dim; q++) {

                    advect_a += 0.;
                    advect_b += grad_phi_i_e_a[p][q] *
                                (d_vconv->rs[q][b][j] * vconv[p] + vconv[q] * d_vconv->rs[p][b][j]);
                    advect_c += 0.;
                  }
                }

                advect_b *= rho * det_J * wt;

                advection = advect_a + advect_b + advect_c;
                advection *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_ADVECTION)];
              }

              source = 0.;

              if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_SOURCE) {
                source = 0.; /* because Mapping jacobian only depends on mesh, for now */
                source *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][LOG2_SOURCE];
              }

              goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, i, j)] +=
                  mass + diffusion + source + advection;
            }
          }
        }

        /*
         * J_r_P
         */
        /* For Lagrangian Mesh, add in the sensitivity to pressure  */
        var = PRESSURE;
        pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];
        if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            diffusion = 0.;
            phi_j = goma_ctx->fem.bf[var]->phi[j];
            for (p = 0; p < dim; p++) {
              for (q = 0; q < dim; q++) {
                grad_phi_i_e_a[p][q] = goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q];

                diffusion += grad_phi_i_e_a[p][q] * dTT_dp[q][p][j];
              }
            }
            diffusion *= -det_J * wt * h3;
            diffusion *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][LOG2_DIFFUSION];

            goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += diffusion;
          }
        }

        /*
         * J_r_c
         */
        var = MASS_FRACTION;
        if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
              /* add inertia of moving mesh */
              advection = 0.;
              /* For Lagrangian Mesh with advection, add in the
                 sensitivity to concentration  */
              if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_ADVECTION) {
                /* Grab this sensitivity from assemble_mesh when you need it */
              }

              diffusion = 0.;
              if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_DIFFUSION) {
                phi_j = goma_ctx->fem.bf[var]->phi[j];
                for (p = 0; p < dim; p++) {
                  for (q = 0; q < dim; q++) {
                    grad_phi_i_e_a[p][q] = goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q];

                    diffusion += grad_phi_i_e_a[p][q] * dTT_dc[q][p][w][j];
                  }
                }
                diffusion *= -det_J * wt * h3;
                diffusion *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][LOG2_DIFFUSION];
              }

              goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, MAX_PROB_VAR + w, i, j)] +=
                  advection + diffusion;
            }
          }
        }

        /*
         * J_r_p_liq
         */
        var = POR_LIQ_PRES;
        pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];
        if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            /* add inertia of moving mesh */
            advection = 0.;
            /* For Lagrangian Mesh with advection, add in the
               sensitivity to concentration  */
            if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_ADVECTION) {
              /* Grab this sensitivity from assemble_mesh when you need it */
            }

            diffusion = 0.;
            if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_DIFFUSION) {
              phi_j = goma_ctx->fem.bf[var]->phi[j];
              for (p = 0; p < dim; p++) {
                for (q = 0; q < dim; q++) {
                  grad_phi_i_e_a[p][q] = goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q];

                  diffusion += grad_phi_i_e_a[p][q] * dTT_dp_liq[q][p][j];
                }
              }
              diffusion *= -det_J * wt * h3;
              diffusion *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][LOG2_DIFFUSION];
            }

            goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += advection + diffusion;
          }
        }

        /*
         * J_r_p_gas
         */
        var = POR_GAS_PRES;
        pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];
        if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            /* add inertia of moving mesh */
            advection = 0.;
            /* For Lagrangian Mesh with advection, add in the
               sensitivity to concentration  */
            if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_ADVECTION) {
              /* Grab this sensitivity from assemble_mesh when you need it */
            }

            diffusion = 0.;
            if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_DIFFUSION) {
              phi_j = goma_ctx->fem.bf[var]->phi[j];
              for (p = 0; p < dim; p++) {
                for (q = 0; q < dim; q++) {
                  grad_phi_i_e_a[p][q] = goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q];

                  diffusion += grad_phi_i_e_a[p][q] * dTT_dp_gas[q][p][j];
                }
              }
              diffusion *= -det_J * wt * h3;
              diffusion *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][LOG2_DIFFUSION];
            }

            goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += advection + diffusion;
          }
        }

        /*
         * J_r_p_gas
         */
        var = POR_POROSITY;
        pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];
        if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            /* add inertia of moving mesh */
            advection = 0.;
            /* For Lagrangian Mesh with advection, add in the
               sensitivity to concentration  */
            if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_ADVECTION) {
              /* Grab this sensitivity from assemble_mesh when you need it */
            }

            diffusion = 0.;
            if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn] & T_DIFFUSION) {
              phi_j = goma_ctx->fem.bf[var]->phi[j];
              for (p = 0; p < dim; p++) {
                for (q = 0; q < dim; q++) {
                  grad_phi_i_e_a[p][q] = goma_ctx->fem.bf[eqn]->grad_phi_e[i][a][p][q];

                  diffusion += grad_phi_i_e_a[p][q] * dTT_dporosity[q][p][j];
                }
              }
              diffusion *= -det_J * wt * h3;
              diffusion *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][LOG2_DIFFUSION];
            }

            goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += advection + diffusion;
          }
        }

        /*
         * J_d_T
         */
        if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][eqn]) {
          var = TEMPERATURE;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
              diffusion = 0.;
              phi_j = goma_ctx->fem.bf[var]->phi[j];
              for (p = 0; p < dim; p++) {
                for (q = 0; q < dim; q++) {
                  diffusion += grad_phi_i_e_a[p][q] * dTT_dT[q][p][j];
                }
              }
              diffusion *= -det_J * wt * h3;
              diffusion *= goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][eqn][LOG2_DIFFUSION];
              goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += advection + diffusion;
            }
          }
        }

      } /* end of loop over equations i  */
    } /* end of loop over equation directions a */
  } /* end of if jacobian */

  return (status);
}

/**************************************************************************************/
/*------------------------------------------------------------*/
/* solid_stress_tensor() -- find TT (stress tensor) for LAGRANGIAN
 * or arbitrary mesh motion
 *
 * Created: 1/16/95 by RAC
 * Modified for solid stress: 2/12/97 by PRS
 *
 */

int solid_stress_tensor(dbl TT[DIM][DIM],
                        dbl dTT_dx[DIM][DIM][DIM][MDE],
                        dbl dTT_drs[DIM][DIM][DIM][MDE],
                        dbl dTT_dp[DIM][DIM][MDE],
                        dbl dTT_dc[DIM][DIM][MAX_CONC][MDE],
                        dbl dTT_dp_liq[DIM][DIM][MDE],
                        dbl dTT_dp_gas[DIM][DIM][MDE],
                        dbl dTT_dporosity[DIM][DIM][MDE],
                        dbl dTT_dT[DIM][DIM][MDE],
                        dbl dTT_dmax_strain[DIM][DIM][MDE],
                        dbl mu,
                        dbl lambda) {
  dbl d_mu_dx[DIM][MDE], d_lambda_dx[DIM][MDE];
  dbl d_mu_drs[DIM][MDE], d_lambda_drs[DIM][MDE];
  int a = 0, b, j, p, q, dim, var, w, v, dofs, err;
  int SPECIES = MAX_VARIABLE_TYPES;

  dbl thermexp = 0., ortho_thermexp = 0;
  dbl speciesexp[MAX_CONC];
  dbl d_thermexp_dx[MAX_VARIABLE_TYPES + MAX_CONC];
  dbl d_ortho_thermexp_dx[MAX_VARIABLE_TYPES + MAX_CONC];
  dbl d_speciesexp_dx[MAX_CONC][MAX_VARIABLE_TYPES + MAX_CONC];
  dbl viscos = 0., dil_viscos = 0;
  dbl d_viscos_dx[MAX_VARIABLE_TYPES + MAX_CONC];
  dbl d_dilviscos_dx[MAX_VARIABLE_TYPES + MAX_CONC];

  dim = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->ielem_dim;

  /*
   * Calculate the lame coefficients if they are not constant. Note, some of the
   * sensitivities of mu and lambda to porous media vars come out through the elc->d_lame
   * structure elements, and hence are not args.
   */

  memset(d_mu_dx, 0, sizeof(double) * DIM * MDE);
  memset(d_lambda_dx, 0, sizeof(double) * DIM * MDE);
  memset(d_mu_drs, 0, sizeof(double) * DIM * MDE);
  memset(d_lambda_drs, 0, sizeof(double) * DIM * MDE);
  memset(d_thermexp_dx, 0, sizeof(double) * (MAX_VARIABLE_TYPES + MAX_CONC));
  memset(d_ortho_thermexp_dx, 0, sizeof(double) * (MAX_VARIABLE_TYPES + MAX_CONC));
  memset(d_speciesexp_dx, 0, sizeof(double) * MAX_CONC * (MAX_VARIABLE_TYPES + MAX_CONC));
  memset(d_viscos_dx, 0, sizeof(double) * (MAX_VARIABLE_TYPES + MAX_CONC));
  memset(d_dilviscos_dx, 0, sizeof(double) * (MAX_VARIABLE_TYPES + MAX_CONC));
  memset(speciesexp, 0, sizeof(double) * MAX_CONC);

  err = load_elastic_properties(goma_ctx->fem.elc_rs, &mu, &lambda, &thermexp, speciesexp, &viscos,
                                &dil_viscos, d_mu_dx, d_lambda_dx, d_thermexp_dx, d_speciesexp_dx,
                                d_viscos_dx, d_dilviscos_dx, &ortho_thermexp, d_ortho_thermexp_dx);
  GOMA_EH(err, " Problem in loading up real-solid elastic constants");

  for (p = 0; p < dim; p++) {
    for (q = 0; q < dim; q++) {
      TT[p][q] = lambda * goma_ctx->fem.fv->volume_strain * delta(p, q) +
                 2. * mu * goma_ctx->fem.fv->strain[p][q];
    }
  }

  /*   add thermo-elasticity terms  */

  if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][R_ENERGY]) {
    for (p = 0; p < dim; p++) {
      for (q = 0; q < dim; q++) {
        TT[p][q] -= 2. * mu * thermexp *
                    (goma_ctx->fem.fv->T - goma_ctx->fem.elc_rs->solid_reference_temp) *
                    delta(p, q);
      }
    }
  }

  if (goma_ctx->fem.af->Assemble_Jacobian) {
    for (p = 0; p < dim; p++) {
      for (q = 0; q < dim; q++) {
        for (b = 0; b < dim; b++) {
          v = MESH_DISPLACEMENT1 + b;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][v]) {
            dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
            for (j = 0; j < dofs; j++) {
              dTT_dx[p][q][b][j] =
                  lambda * goma_ctx->fem.fv->d_volume_strain_dx[b][j] * delta(p, q) +
                  2. * mu * goma_ctx->fem.fv->d_strain_dx[p][q][b][j];

              dTT_dx[p][q][b][j] +=
                  d_lambda_dx[b][j] * goma_ctx->fem.fv->volume_strain * delta(p, q) +
                  2. * d_mu_dx[b][j] * goma_ctx->fem.fv->strain[p][q];
              if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][R_ENERGY]) {
                dTT_dx[p][q][b][j] -=
                    2. * d_mu_dx[b][j] * thermexp *
                    (goma_ctx->fem.fv->T - goma_ctx->fem.elc_rs->solid_reference_temp) *
                    delta(p, q);
              }
            }
          }
          v = SOLID_DISPLACEMENT1 + b;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][v]) {
            dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
            for (j = 0; j < dofs; j++) {
              dTT_drs[p][q][b][j] =
                  lambda * goma_ctx->fem.fv->d_volume_strain_drs[b][j] * delta(p, q) +
                  2. * mu * goma_ctx->fem.fv->d_strain_drs[p][q][b][j];

              dTT_drs[p][q][b][j] +=
                  d_lambda_drs[b][j] * goma_ctx->fem.fv->volume_strain * delta(p, q) +
                  2. * d_mu_drs[b][j] * goma_ctx->fem.fv->strain[p][q];
              if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][R_ENERGY]) {
                dTT_drs[p][q][b][j] -=
                    2. * d_mu_drs[b][j] * thermexp *
                    (goma_ctx->fem.fv->T - goma_ctx->fem.elc_rs->solid_reference_temp) *
                    delta(p, q);
              }
            }
          }
        }
        /*  Temperature sensitivities here */
        v = TEMPERATURE;
        if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][v]) {
          dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];

          for (j = 0; j < dofs; j++) {
            /*if no temperature dependence d_lame_mu is zero because we initialize in
             * set_mp_to_unity */
            dTT_dT[p][q][j] +=
                (2. * goma_ctx->fem.elc->d_lame_mu[TEMPERATURE] * goma_ctx->fem.fv->strain[p][q]) *
                goma_ctx->fem.bf[v]->phi[j];
          }

          for (j = 0; j < dofs; j++) {
            dTT_dT[p][q][j] = -2. * mu * thermexp * goma_ctx->fem.bf[v]->phi[j] * delta(p, q);
          }
        }
        /*  max_strain sensitivities here */
        v = MAX_STRAIN;
        if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][v]) {
          dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];

          if (goma_ctx->fem.elc->lame_mu_model == TABLE) {
            for (j = 0; j < dofs; j++) {
              dTT_dmax_strain[p][q][j] += goma_ctx->fem.fv->volume_strain * delta(p, q) *
                                          goma_ctx->fem.elc->d_lame_lambda[v] *
                                          goma_ctx->fem.bf[v]->phi[j];
              dTT_dmax_strain[p][q][j] += 2.0 * goma_ctx->fem.fv->strain[p][q] *
                                          goma_ctx->fem.elc->d_lame_mu[v] *
                                          goma_ctx->fem.bf[v]->phi[j];
            }
          }
        }
      }
    }
    /* PRESSURE, MASS_FRACTION, and porous sensitivities are taken care of later */
  }

  /*
   * add on the pressure contribution to the stress tensor for the porous or
   * incompressible formulations
   */
  /* now, add in pressure due to swelling for incompressible lagrangian
   * mesh motion */
  var = PRESSURE;
  if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var] && !goma_ctx->fem.mp->PorousMediaType) {
    if (goma_ctx->fem.cr->RealSolidFluxModel == INCOMP_PSTRAIN ||
        goma_ctx->fem.cr->RealSolidFluxModel == INCOMP_PSTRESS ||
        goma_ctx->fem.cr->RealSolidFluxModel == INCOMP_3D) {
      for (a = 0; a < dim; a++) {
        TT[a][a] -= goma_ctx->fem.fv->P;
      }
    }
  }

  /* pressure force is the pressure of each phase times it's volume fraction */
  if (goma_ctx->fem.mp->PorousMediaType == POROUS_UNSATURATED ||
      goma_ctx->fem.mp->PorousMediaType == POROUS_SATURATED ||
      goma_ctx->fem.mp->PorousMediaType == POROUS_TWO_PHASE) {
    for (a = 0; a < dim; a++) {
      if (goma_ctx->fem.mp->CapStress == COMPRESSIBLE) {
        GOMA_EH(GOMA_ERROR, "Need to reconcile the COMPRESSIBLE model pressures.  Not ready yet");
        /* Compressible model of effective stress law with
         *  partially saturated media from Zienkeivicz and Garg and Nur */
        if (goma_ctx->fem.elc_rs->lame_lambda_model != POWER_LAW)
          GOMA_WH(-1, "Effective stress law may be missing constant");
        TT[a][a] -= (1 - (1 - goma_ctx->fem.mp->porosity) * goma_ctx->fem.elc_rs->lame_lambda /
                             goma_ctx->fem.elc_rs->u_lambda[0]) *
                    goma_ctx->fem.mp->saturation * goma_ctx->fem.fv->p_liq;
        if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][POR_GAS_PRES]) {
          TT[a][a] -= (1 - (1 - goma_ctx->fem.mp->porosity) * goma_ctx->fem.elc_rs->lame_lambda /
                               goma_ctx->fem.elc_rs->u_lambda[0]) *
                      (1. - goma_ctx->fem.mp->saturation) * goma_ctx->fem.fv->p_gas;
        }

      } else if (goma_ctx->fem.mp->CapStress == PARTIALLY_WETTING) {
        GOMA_EH(GOMA_ERROR,
                "See Lagragian case, this effective stress may not be correct!.  Not ready yet");
        TT[a][a] -= goma_ctx->fem.mp->saturation * goma_ctx->fem.fv->p_liq;
        if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][POR_GAS_PRES])
          TT[a][a] -= (1. - goma_ctx->fem.mp->saturation) * goma_ctx->fem.fv->p_gas;

      } else if (goma_ctx->fem.mp->CapStress == WETTING) {
        GOMA_EH(GOMA_ERROR, "Need to reconcile the WETTING model pressures.  Not ready yet");
        /* If liquid is wetting, so that all surfaces are covered
           by a thin layer of liquid */
        TT[a][a] -= (1 - goma_ctx->fem.mp->porosity * (1. - goma_ctx->fem.mp->saturation)) *
                    goma_ctx->fem.fv->p_liq;
        if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][POR_GAS_PRES])
          TT[a][a] -= goma_ctx->fem.mp->porosity * (1. - goma_ctx->fem.mp->saturation) *
                      goma_ctx->fem.fv->p_gas;
      } else if (goma_ctx->fem.mp->PorousMediaType == POROUS_SATURATED) {
        TT[a][a] -= goma_ctx->fem.fv->p_liq;

      } else
        GOMA_EH(GOMA_ERROR, "no way to put liquid stress into porous matrix");
    }
  }

  if (goma_ctx->fem.af->Assemble_Jacobian) {
    var = PRESSURE;
    if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
      if (goma_ctx->fem.cr->RealSolidFluxModel == INCOMP_PSTRESS ||
          goma_ctx->fem.cr->RealSolidFluxModel == HOOKEAN_PSTRESS) {
        for (p = 0; p < dim; p++) {
          for (q = 0; q < dim; q++) {
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
              /* didn't add this sensitivity in before - add it now */
              dTT_dp[p][q][j] = 2. * mu * goma_ctx->fem.fv->d_strain_dp[p][q][j] +
                                lambda * goma_ctx->fem.fv->d_volume_strain_dp[j] * delta(p, q);
            }
          }
        }
      }

      if ((goma_ctx->fem.cr->RealSolidFluxModel == INCOMP_PSTRAIN ||
           goma_ctx->fem.cr->RealSolidFluxModel == INCOMP_PSTRESS ||
           goma_ctx->fem.cr->RealSolidFluxModel == INCOMP_3D) &&
          !goma_ctx->fem.mp->PorousMediaType) {
        for (a = 0; a < dim; a++) {
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            dTT_dp[a][a][j] -= goma_ctx->fem.bf[var]->phi[j];
          }
        }
      }
    }

    /* calculate sensitivity of stress tensor due to concentration dependent
       elastic and bulk modulus */

    var = MASS_FRACTION;
    if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {

      /* sensitivity of total stress to species concentrations (or pressures)
         - first calculate contribution from sensitivity of elastic moduli */
      for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
        for (a = 0; a < dim; a++) {
          for (b = 0; b < dim; b++) {
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
              /* didn't add this in before - add it in now */
              dTT_dc[a][b][w][j] += goma_ctx->fem.bf[var]->phi[j] *
                                    (goma_ctx->fem.elc_rs->d_lame_lambda[SPECIES + w] *
                                         goma_ctx->fem.fv->volume_strain * delta(a, b) +
                                     2. * goma_ctx->fem.elc_rs->d_lame_mu[SPECIES + w] *
                                         goma_ctx->fem.fv->strain[a][b]);
            }
          }
        }
      }
    }

    /* calculate sensitivity of stress tensor due to
       porous media vars dependent elastic and bulk modulus */

    var = POR_LIQ_PRES; /*PRS:  PLEASE Check this against that in mesh_stress_tensor*/
    if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {

      /* sensitivity of total stress to porous media vars
         - first calculate contribution from sensitivity of elastic moduli */

      /* in porous media, stress has contribution from phase pressures */
      if (goma_ctx->fem.mp->PorousMediaType == POROUS_UNSATURATED ||
          goma_ctx->fem.mp->PorousMediaType == POROUS_SATURATED ||
          goma_ctx->fem.mp->PorousMediaType == POROUS_TWO_PHASE) {

        for (p = 0; p < dim; p++) {
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            if (goma_ctx->fem.mp->CapStress == COMPRESSIBLE) {
              dTT_dp_liq[p][p][j] -=
                  goma_ctx->fem.bf[var]->phi[j] *
                  (1 - (1 - goma_ctx->fem.mp->porosity) * goma_ctx->fem.elc_rs->lame_lambda /
                           goma_ctx->fem.elc_rs->u_lambda[0]) *
                  (goma_ctx->fem.mp->saturation +
                   goma_ctx->fem.mp->d_saturation[POR_LIQ_PRES] * goma_ctx->fem.fv->p_liq);

              if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][POR_GAS_PRES]) {
                dTT_dp_gas[p][p][j] -=
                    goma_ctx->fem.bf[var]->phi[j] *
                    (1 - (1 - goma_ctx->fem.mp->porosity) * goma_ctx->fem.elc_rs->lame_lambda /
                             goma_ctx->fem.elc_rs->u_lambda[0]) *
                    ((1. - goma_ctx->fem.mp->saturation) -
                     goma_ctx->fem.mp->d_saturation[POR_GAS_PRES] * goma_ctx->fem.fv->p_gas);
              }
              if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][POR_POROSITY]) {
                dTT_dporosity[p][p][j] -=
                    goma_ctx->fem.bf[var]->phi[j] *
                    (goma_ctx->fem.mp->d_porosity[POR_POROSITY] *
                         goma_ctx->fem.elc_rs->lame_lambda / goma_ctx->fem.elc_rs->u_lambda[0] *
                         goma_ctx->fem.mp->saturation -
                     (1 - goma_ctx->fem.mp->porosity) *
                         goma_ctx->fem.elc_rs->d_lame_lambda[POR_POROSITY] /
                         goma_ctx->fem.elc_rs->u_lambda[0] * goma_ctx->fem.mp->saturation +
                     (1 - (1 - goma_ctx->fem.mp->porosity) * goma_ctx->fem.elc_rs->lame_lambda /
                              goma_ctx->fem.elc_rs->u_lambda[0]) *
                         goma_ctx->fem.mp->d_saturation[POR_POROSITY]) *
                    goma_ctx->fem.fv->p_liq;
              }
              if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][POR_GAS_PRES] &&
                  goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][POR_POROSITY]) {
                dTT_dporosity[p][p][j] -=
                    goma_ctx->fem.bf[var]->phi[j] *
                    (goma_ctx->fem.mp->d_porosity[POR_POROSITY] *
                         goma_ctx->fem.elc_rs->lame_lambda / goma_ctx->fem.elc_rs->u_lambda[0] *
                         (1 - goma_ctx->fem.mp->saturation) -
                     (1 - (1 - goma_ctx->fem.mp->porosity) * goma_ctx->fem.elc_rs->lame_lambda /
                              goma_ctx->fem.elc_rs->u_lambda[0]) *
                         goma_ctx->fem.mp->d_saturation[POR_POROSITY]) *
                    goma_ctx->fem.fv->p_gas;
              }

            } else if (goma_ctx->fem.mp->CapStress == PARTIALLY_WETTING) {
              dTT_dp_liq[p][p][j] -=
                  goma_ctx->fem.bf[var]->phi[j] *
                  (goma_ctx->fem.mp->saturation +
                   goma_ctx->fem.mp->d_saturation[POR_LIQ_PRES] * goma_ctx->fem.fv->p_liq);
              if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][POR_GAS_PRES]) {
                dTT_dp_gas[p][p][j] -= goma_ctx->fem.bf[var]->phi[j] *
                                       ((1. - goma_ctx->fem.mp->saturation) +
                                        goma_ctx->fem.mp->d_saturation[POR_GAS_PRES] *
                                            (goma_ctx->fem.fv->p_liq - goma_ctx->fem.fv->p_gas));
              }
              if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][POR_POROSITY]) {
                dTT_dporosity[p][p][j] -= goma_ctx->fem.bf[var]->phi[j] *
                                          goma_ctx->fem.mp->d_saturation[POR_POROSITY] *
                                          goma_ctx->fem.fv->p_liq;
              }
              if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][POR_POROSITY] &&
                  goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][POR_GAS_PRES]) {
                dTT_dporosity[p][p][j] += goma_ctx->fem.bf[var]->phi[j] *
                                          goma_ctx->fem.mp->d_saturation[POR_POROSITY] *
                                          goma_ctx->fem.fv->p_gas;
              }

            } else if (goma_ctx->fem.mp->CapStress == WETTING) {
              /* If liquid is wetting, so that all surfaces are covered
                 by a thin layer of liquid */
              dTT_dp_liq[p][p][j] -=
                  goma_ctx->fem.bf[var]->phi[j] *
                  ((1 - goma_ctx->fem.mp->porosity * (1. - goma_ctx->fem.mp->saturation)) +
                   goma_ctx->fem.mp->porosity * goma_ctx->fem.mp->d_saturation[POR_LIQ_PRES] *
                       goma_ctx->fem.fv->p_liq);
              if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][POR_GAS_PRES]) {
                dTT_dp_gas[p][p][j] -=
                    goma_ctx->fem.bf[var]->phi[j] * goma_ctx->fem.mp->porosity *
                    ((1. - goma_ctx->fem.mp->saturation) -
                     goma_ctx->fem.mp->d_saturation[POR_GAS_PRES] * goma_ctx->fem.fv->p_gas);
              }
              if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][POR_POROSITY]) {
                dTT_dporosity[p][p][j] -=
                    goma_ctx->fem.bf[var]->phi[j] *
                    (-goma_ctx->fem.mp->d_porosity[POR_POROSITY] *
                         (1. - goma_ctx->fem.mp->saturation) +
                     goma_ctx->fem.mp->porosity * goma_ctx->fem.mp->d_saturation[POR_POROSITY]) *
                    goma_ctx->fem.fv->p_liq;
              }
              if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][POR_POROSITY] &&
                  goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][POR_GAS_PRES]) {
                dTT_dporosity[p][p][j] -=
                    goma_ctx->fem.bf[var]->phi[j] *
                    (goma_ctx->fem.mp->d_porosity[POR_POROSITY] *
                         (1. - goma_ctx->fem.mp->saturation) -
                     goma_ctx->fem.mp->porosity * goma_ctx->fem.mp->d_saturation[POR_POROSITY]) *
                    goma_ctx->fem.fv->p_gas;
              }

            } else if (goma_ctx->fem.mp->PorousMediaType == POROUS_SATURATED) {
              dTT_dp_liq[a][a][j] -= goma_ctx->fem.bf[var]->phi[j];
            }
          }
        }
      }
    }

  } /* end of if Jacobian */

  return (0);
} /* end of solid_stress_tensor()*/

/*****************************************************************************/
/* Counterpart to belly_flop (mm_fill_solid.c) for real solid case */
/*****************************************************************************/

int belly_flop_rs(dbl mu) /* elastic modulus - (for plane stress case) */
{
  int status;
  int i, j, k, p, q, a, b;
  int dim;
  int mdof;

  dbl factor;                /* coefficients in deviatoric strain matrix */
  dbl deform_grad[DIM][DIM]; /* deformation gradient for nonlinear elasticity
                                is d(deformation)/d(initial mesh coord) + identity */
  dbl invdeform_grad[DIM][DIM];
  dbl d_invdeform_grad_dx[DIM][DIM][DIM][MDE];
  dbl d_invdeform_grad_drs[DIM][DIM][DIM][MDE];
  dbl d_deform_grad_dx[DIM][DIM][DIM][MDE];
  dbl d_deform_grad_drs[DIM][DIM][DIM][MDE];
  dbl grad_d_rs[DIM][DIM];                /* displacement gradient*/
  dbl d_grad_d[DIM][DIM][DIM][MDE];       /* mesh displacement gradient wrt to mesh*/
  dbl d_grad_d_rs[DIM][DIM][DIM][MDE];    /* solid displacement gradient sensitivity wrt mesh */
  dbl d_grad_d_rs_rs[DIM][DIM][DIM][MDE]; /* solid displacement gradient sensitivity
                                             wrt solid displacement*/
  dbl det2d;                              /* determinant of 2D deformation gradient tensor */
  dbl ddet2d_dx[DIM][MDE];                /* sensitivity */
  dbl ddet2d_drs[DIM][MDE];               /* sensitivity */
  dbl cauchy_green[DIM][DIM];             /* strain tensor without division by determinant, etc. */
  dbl d_cauchy_green_dx[DIM][DIM][DIM][MDE];  /* sensitivity */
  dbl d_cauchy_green_drs[DIM][DIM][DIM][MDE]; /* sensitivity */

  status = 0;

  dim = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->ielem_dim;
  mdof = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[SOLID_DISPLACEMENT1];
  /*******************************************************************************/
  /* load up the displacement gradient and it's sensitivities                    */
  /*******************************************************************************/
  /* Initialize arrays used here */
  memset(deform_grad, 0, sizeof(double) * DIM * DIM);
  memset(d_deform_grad_dx, 0, sizeof(double) * DIM * DIM * DIM * MDE);
  memset(d_deform_grad_drs, 0, sizeof(double) * DIM * DIM * DIM * MDE);
  memset(grad_d_rs, 0, sizeof(double) * DIM * DIM);
  memset(cauchy_green, 0, sizeof(double) * DIM * DIM);
  memset(goma_ctx->fem.fv->strain, 0, sizeof(double) * DIM * DIM);
  memset(goma_ctx->fem.fv->deform_grad_rs, 0, sizeof(double) * DIM * DIM);
  if (goma_ctx->fem.af->Assemble_Jacobian) {
    memset(d_grad_d, 0, sizeof(double) * DIM * DIM * DIM * MDE);
    memset(d_grad_d_rs, 0, sizeof(double) * DIM * DIM * DIM * MDE);
    memset(d_cauchy_green_dx, 0, sizeof(double) * DIM * DIM * DIM * MDE);
    memset(d_cauchy_green_drs, 0, sizeof(double) * DIM * DIM * DIM * MDE);
    memset(ddet2d_dx, 0, sizeof(double) * DIM * MDE);
    memset(ddet2d_drs, 0, sizeof(double) * DIM * MDE);
    memset(goma_ctx->fem.fv->d_volume_change_dx, 0, sizeof(double) * DIM * MDE);
    memset(goma_ctx->fem.fv->d_volume_change_drs, 0, sizeof(double) * DIM * MDE);
    memset(goma_ctx->fem.fv->d_volume_strain_dx, 0, sizeof(double) * DIM * MDE);
    memset(goma_ctx->fem.fv->d_volume_strain_drs, 0, sizeof(double) * DIM * MDE);
    memset(goma_ctx->fem.fv->d_volume_change_dp, 0, sizeof(double) * MDE);
    memset(goma_ctx->fem.fv->d_volume_strain_dp, 0, sizeof(double) * MDE);
    memset(goma_ctx->fem.fv->d_strain_dx, 0, sizeof(double) * DIM * DIM * DIM * MDE);
    memset(goma_ctx->fem.fv->d_strain_drs, 0, sizeof(double) * DIM * DIM * DIM * MDE);
    memset(goma_ctx->fem.fv->d_strain_dp, 0, sizeof(double) * DIM * DIM * MDE);
    memset(goma_ctx->fem.fv->d_deform_grad_rs_dx, 0, sizeof(double) * DIM * DIM * DIM * MDE);
    memset(goma_ctx->fem.fv->d_deform_grad_rs_drs, 0, sizeof(double) * DIM * DIM * DIM * MDE);
  }

  /* First, if this is an pure Eulerian solid mechanics problem, we have
   * called belly_flop to get the mesh def-grad tensor, hence it is
   * zero.   Actually, it should be the diagonal tensor for the case of no
   * mesh deformation.  So make it as such
   */
  if (goma_ctx->fem.pd->TimeIntegration != STEADY &&
      goma_ctx->fem.pd->etm[goma_ctx->fem.pg->imtrx][R_SOLID1][(LOG2_MASS)] &&
      !goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][R_MESH1]) {
    for (p = 0; p < dim; p++) {
      for (q = 0; q < dim; q++) {
        goma_ctx->fem.fv->deform_grad[p][q] = delta(p, q);
      }
    }
  }
  /*
   * grad(d_rs)
   */

  /* Use Coordinates of Main Problem */
  for (p = 0; p < dim; p++) {
    for (q = 0; q < dim; q++) {
      grad_d_rs[p][q] = goma_ctx->fem.fv->grad_d_rs[p][q];
    }
  }
  if (goma_ctx->fem.af->Assemble_Jacobian) {
    for (p = 0; p < dim; p++) {
      for (q = 0; q < dim; q++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[MESH_DISPLACEMENT1 + b];
               j++) {
            d_grad_d_rs[p][q][b][j] = goma_ctx->fem.fv->d_grad_d_rs_dmesh[p][q][b][j];
          }
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[SOLID_DISPLACEMENT1 + b];
               j++) {
            d_grad_d_rs_rs[p][q][b][j] = goma_ctx->fem.bf[R_SOLID1]->grad_phi[j][p] * delta(q, b);
          }
        }
      }
    }
  } /* end of Assemble_Jacobian */

  /*******************************************************************************/
  /* calculate Lagrangian deformation gradient and it's invariants */
  /* First invariant - trace */
  /* Second invariant - shearing */
  /* Third invariant - volume change */
  /*******************************************************************************/
  if (goma_ctx->fem.cr->RealSolidFluxModel == LINEAR) {
    /*
     * N.B. Lagrangian and Eulerian displacement Gradients are the same.  In this
     * case the grad_d_rs is WRT the spatial deformed mesh, but it strictly should
     * be WRT the material rs coordinates.   HOWEVER, for small deformations, this
     * is a good approximation
     */
    for (p = 0; p < dim; p++) {
      for (q = 0; q < dim; q++) {
        goma_ctx->fem.fv->deform_grad_rs[p][q] = delta(p, q) + grad_d_rs[p][q];
      }
    }

    if (goma_ctx->fem.af->Assemble_Jacobian) {
      for (p = 0; p < dim; p++) {
        for (q = 0; q < dim; q++) {
          for (b = 0; b < dim; b++) {
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[MESH_DISPLACEMENT1 + b];
                 j++) {
              goma_ctx->fem.fv->d_deform_grad_rs_dx[p][q][b][j] = d_grad_d_rs[p][q][b][j];
            }
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[SOLID_DISPLACEMENT1 + b];
                 j++) {
              goma_ctx->fem.fv->d_deform_grad_rs_drs[p][q][b][j] = d_grad_d_rs_rs[p][q][b][j];
            }
          }
        }
      }
    }
  } else { /* NONLINEAR MESH */
    /*
     * Lagrangian and Eulerian Deformation Gradients are inverses
     * Actually, we will take a different approach here than in the
     * Lagrangian case.  We actually will compute the deform_grad
     * of the material first and then invert.  This is necessary to
     * transform material gradient into the easily computable spatial
     * gradient.  BUT, it does require the deformation gradient of the
     * mesh to be available.   Remember, this is the anti-BSL convention
     * with the first subscript being the denominator:
     * viz. dx_q/dX_p = deform_grad[p][q]
     *
     * Also NOTE:  Unlike belly_flop, we use fv->deform_grad which is
     * computed there in belly_flop as the Lagrangian tensor.  Interestingly, it is
     * not used there as it is supplanted by the linearized small strain
     * theory.   We use it here, though.
     */
    for (a = 0; a < dim; a++) {
      for (p = 0; p < dim; p++) {
        deform_grad[p][a] += delta(p, a);

        for (q = 0; q < dim; q++) {
          deform_grad[p][a] += grad_d_rs[p][q] * goma_ctx->fem.fv->deform_grad[q][a];
        }
      }
    }

    if (goma_ctx->fem.af->Assemble_Jacobian) {
      for (p = 0; p < dim; p++) {
        for (a = 0; a < dim; a++) {
          for (b = 0; b < dim; b++) {
            for (q = 0; q < dim; q++) {
              for (j = 0;
                   j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[MESH_DISPLACEMENT1 + b];
                   j++) {
                /* Remember, this is the deformation gradient
                 * of the material and it uses, as shown
                 *  above the deformation gradient of the mesh
                 */
                d_deform_grad_dx[p][a][b][j] +=
                    grad_d_rs[p][q] * goma_ctx->fem.fv->d_deform_grad_dx[q][a][b][j] +
                    d_grad_d_rs[p][q][b][j] * goma_ctx->fem.fv->deform_grad[q][a];
              }
              for (j = 0;
                   j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[SOLID_DISPLACEMENT1 + b];
                   j++) {
                d_deform_grad_drs[p][a][b][j] +=
                    d_grad_d_rs_rs[p][q][b][j] * goma_ctx->fem.fv->deform_grad[q][a];
              }
            }
          }
        }
      }
    }

    /** Ok we are going to be a little redundant here for
        convenience on the first implementation.  The deformation
        gradient wrt the real solid  has sensitivities wrt to
        both d and d_rs, so we call this routine twice with the
        correct input, for now. To fix you should send in a
        flag to route your way through */

    invert_tensor(deform_grad, invdeform_grad, dim, d_deform_grad_dx, d_invdeform_grad_dx,
                  goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[SOLID_DISPLACEMENT1],
                  goma_ctx->fem.af->Assemble_Jacobian);
    invert_tensor(deform_grad, invdeform_grad, dim, d_deform_grad_drs, d_invdeform_grad_drs,
                  goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[SOLID_DISPLACEMENT1],
                  goma_ctx->fem.af->Assemble_Jacobian);

    /*Also, so we have them, load up the fv structure global variable of the
     *material deform_grad
     */

    for (p = 0; p < dim; p++) {
      for (a = 0; a < dim; a++) {
        goma_ctx->fem.fv->deform_grad_rs[p][a] = deform_grad[p][a];

        if (goma_ctx->fem.af->Assemble_Jacobian) {
          for (b = 0; b < dim; b++) {
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[MESH_DISPLACEMENT1 + b];
                 j++) {
              goma_ctx->fem.fv->d_deform_grad_rs_dx[p][a][b][j] = d_deform_grad_dx[p][a][b][j];
            }
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[SOLID_DISPLACEMENT1 + b];
                 j++) {
              goma_ctx->fem.fv->d_deform_grad_rs_drs[p][a][b][j] = d_deform_grad_drs[p][a][b][j];
            }
          }
        }
      }
    }
  }

  /*******************************************************************************/
  /* calculate basic Cauchy-Green strain tensor (F.F_T - I)
   * and it's sensitivity */
  /*******************************************************************************/

  /* Start with contribution from linear elasticity */
  for (p = 0; p < dim; p++) {
    for (q = 0; q < dim; q++) {
      cauchy_green[p][q] = -0.5 * delta(p, q);
    }
  }
  /* add on nonlinear term to Eulerian Strain Tensor */
  if (goma_ctx->fem.cr->RealSolidFluxModel != LINEAR) {
    for (p = 0; p < dim; p++) {
      for (q = 0; q < dim; q++) {
        for (a = 0; a < dim; a++) {
          /* dot product between displacement gradient and
           * it's transpose - note that there are two ways to
           * do this
           * the right way sums d(d_j)/d(x_m) d(d_j)/d(x_k)
           * over j to get B_m_k
           */
          cauchy_green[p][q] += 0.5 * deform_grad[p][a] * deform_grad[q][a];
        }
      }
    }
  }

  if (goma_ctx->fem.af->Assemble_Jacobian) {
    for (p = 0; p < dim; p++) {
      for (q = 0; q < dim; q++) {
        for (a = 0; a < dim; a++) {
          for (b = 0; b < dim; b++) {
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[MESH_DISPLACEMENT1 + b];
                 j++) {
              d_cauchy_green_dx[p][q][b][j] +=
                  0.5 * (d_deform_grad_dx[p][a][b][j] * deform_grad[q][a] +
                         deform_grad[p][a] * d_deform_grad_dx[q][a][b][j]);
            }
          }
        }
      }
    }

    for (p = 0; p < dim; p++) {
      for (q = 0; q < dim; q++) {
        for (a = 0; a < dim; a++) {
          for (b = 0; b < dim; b++) {
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[SOLID_DISPLACEMENT1 + b];
                 j++) {
              d_cauchy_green_drs[p][q][b][j] +=
                  0.5 * (d_deform_grad_drs[p][a][b][j] * deform_grad[q][a] +
                         deform_grad[p][a] * d_deform_grad_drs[q][a][b][j]);
            }
          }
        }
      }
    }
  }

  /*******************************************************************************/
  /* calculate the volume change needed in Neo-Hookean Constitutive Law
   * (V - Vo)/Vo */
  /*******************************************************************************/
  goma_ctx->fem.fv->volume_change = 0.;
  goma_ctx->fem.fv->volume_strain = 0.;

  if (goma_ctx->fem.cr->RealSolidFluxModel == LINEAR) {
    /*
     * Find the trace of the deformation gradient, which in linear elasticity is the
     * volume change
     */
    goma_ctx->fem.fv->volume_change = 1.;
    goma_ctx->fem.fv->volume_strain = 0.;
    for (p = 0; p < dim; p++) {
      goma_ctx->fem.fv->volume_change += cauchy_green[p][p];
      goma_ctx->fem.fv->volume_strain += cauchy_green[p][p];
    }

    if (goma_ctx->fem.af->Assemble_Jacobian) {
      for (p = 0; p < dim; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[MESH_DISPLACEMENT1 + b];
               j++) {
            goma_ctx->fem.fv->d_volume_change_dx[b][j] += d_cauchy_green_dx[p][p][b][j];
            goma_ctx->fem.fv->d_volume_strain_dx[b][j] += d_cauchy_green_dx[p][p][b][j];
          }
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[SOLID_DISPLACEMENT1 + b];
               j++) {
            goma_ctx->fem.fv->d_volume_change_drs[b][j] += d_cauchy_green_drs[p][p][b][j];
            goma_ctx->fem.fv->d_volume_strain_drs[b][j] += d_cauchy_green_drs[p][p][b][j];
          }
        }
      }
    }
  } else { /* Non-Linear Mesh Motion */
    /*
     * volume change is the determinant of the Lagrangian deformation gradient tensor
     * or reciprocal of determinant of Eulerian deformation gradient tensor. We just
     * computed the Lagrangian material deformation tensor deform_grad.   The Eulerian
     * is its inverse.
     */

    switch (dim) {
    case 1:
      goma_ctx->fem.fv->volume_change = deform_grad[0][0];
      goma_ctx->fem.fv->volume_strain = goma_ctx->fem.fv->volume_change - 1.;
      if (goma_ctx->fem.af->Assemble_Jacobian) {
        for (i = 0; i < dim; i++) {
          for (k = 0; k < mdof; k++) {
            goma_ctx->fem.fv->d_volume_change_dx[i][k] = d_deform_grad_dx[0][0][i][k];
            goma_ctx->fem.fv->d_volume_strain_dx[i][k] = goma_ctx->fem.fv->d_volume_change_dx[i][k];
            goma_ctx->fem.fv->d_volume_change_drs[i][k] = d_deform_grad_drs[0][0][i][k];
            goma_ctx->fem.fv->d_volume_strain_drs[i][k] =
                goma_ctx->fem.fv->d_volume_change_drs[i][k];
          }
        }
      }
      break;

    case 2:
      /* find determinant of 2-d deformation gradient (note this is not the volume change, that
         is the determinant of the 3-d deformation gradient which is here approximated by plane
         strain or plane stress for 2-d) */

      det2d = (deform_grad[0][0] * deform_grad[1][1] - deform_grad[0][1] * deform_grad[1][0]);

      /* escape/warn if element has inverted */
      if ((det2d <= 0.) && (goma_ctx->debug_flag >= 0)) {
#ifdef PARALLEL
        fprintf(stderr, "P_%d: Volume change  %f\n", goma_ctx->proc_id, det2d);
#else
        fprintf(stderr, "Volume change  %f\n", det2d);
#endif
        for (i = 0; i < dim; i++) {
#ifdef PARALLEL
          fprintf(stderr, "P_%d: Deformation Gradient of real solid  %f  %f\n", goma_ctx->proc_id,
                  deform_grad[i][0], deform_grad[i][1]);
#else
          fprintf(stderr, "Deformation Gradient of real solid  %f  %f\n", deform_grad[i][0],
                  deform_grad[i][1]);
#endif
        }
        goma_ctx->solver.neg_elem_volume = TRUE;
        if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][R_MESH1])
          return (2);
      }
      if (det2d <= 0.) {
        goma_ctx->solver.neg_elem_volume = TRUE;
        if (goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][R_MESH1])
          return (2);
      }

      if (goma_ctx->fem.af->Assemble_Jacobian) {
        for (i = 0; i < dim; i++) {
          for (k = 0; k < mdof; k++) {
            ddet2d_dx[i][k] = deform_grad[0][0] * d_deform_grad_dx[1][1][i][k] +
                              d_deform_grad_dx[0][0][i][k] * deform_grad[1][1] -
                              deform_grad[0][1] * d_deform_grad_dx[1][0][i][k] -
                              d_deform_grad_dx[0][1][i][k] * deform_grad[1][0];

            ddet2d_drs[i][k] = deform_grad[0][0] * d_deform_grad_drs[1][1][i][k] +
                               d_deform_grad_drs[0][0][i][k] * deform_grad[1][1] -
                               deform_grad[0][1] * d_deform_grad_drs[1][0][i][k] -
                               d_deform_grad_drs[0][1][i][k] * deform_grad[1][0];
          }
        }
      }

      /* PLANE STRAIN CASES */
      if (goma_ctx->fem.cr->RealSolidFluxModel == NONLINEAR ||
          goma_ctx->fem.cr->RealSolidFluxModel == INCOMP_PSTRAIN ||
          goma_ctx->fem.cr->RealSolidFluxModel == HOOKEAN_PSTRAIN) {
        goma_ctx->fem.fv->volume_change = det2d;
        goma_ctx->fem.fv->volume_strain = 3. * (pow(det2d, 1. / 3.) - 1.);

        if (goma_ctx->fem.af->Assemble_Jacobian) {
          for (i = 0; i < dim; i++) {
            for (k = 0; k < mdof; k++) {
              goma_ctx->fem.fv->d_volume_change_dx[i][k] = ddet2d_dx[i][k];
              goma_ctx->fem.fv->d_volume_strain_dx[i][k] = ddet2d_dx[i][k] * pow(det2d, -2. / 3.);
              goma_ctx->fem.fv->d_volume_change_drs[i][k] = ddet2d_drs[i][k];
              goma_ctx->fem.fv->d_volume_strain_drs[i][k] = ddet2d_drs[i][k] * pow(det2d, -2. / 3.);
            }
          }
        }
      }
      /* PLANE STRESS CASES */
      else if (goma_ctx->fem.cr->RealSolidFluxModel == INCOMP_PSTRESS) {
        GOMA_EH(GOMA_ERROR, "need to fix Plane Stress cases");
      }
      break;
    case 3:
      goma_ctx->fem.fv->volume_change =
          deform_grad[0][0] *
              (deform_grad[1][1] * deform_grad[2][2] - deform_grad[1][2] * deform_grad[2][1]) -
          deform_grad[0][1] *
              (deform_grad[1][0] * deform_grad[2][2] - deform_grad[2][0] * deform_grad[1][2]) +
          deform_grad[0][2] *
              (deform_grad[1][0] * deform_grad[2][1] - deform_grad[2][0] * deform_grad[1][1]);

      goma_ctx->fem.fv->volume_strain = 3. * (pow(goma_ctx->fem.fv->volume_change, 1. / 3.) - 1.);

      if (goma_ctx->fem.af->Assemble_Jacobian)

      /* I think some key terms are missing PRS 6/16/99 */
      {
        for (i = 0; i < dim; i++) {
          for (k = 0; k < mdof; k++) {
            goma_ctx->fem.fv->d_volume_change_dx[i][k] =
                (d_deform_grad_dx[0][0][i][k] * (deform_grad[1][1] * deform_grad[2][2] -
                                                 deform_grad[1][2] * deform_grad[2][1]) -
                 d_deform_grad_dx[0][1][i][k] * (deform_grad[1][0] * deform_grad[2][2] -
                                                 deform_grad[2][0] * deform_grad[1][2]) +
                 d_deform_grad_dx[0][2][i][k] * (deform_grad[1][0] * deform_grad[2][1] -
                                                 deform_grad[2][0] * deform_grad[1][1]) +
                 deform_grad[0][0] * (d_deform_grad_dx[1][1][i][k] * deform_grad[2][2] -
                                      d_deform_grad_dx[1][2][i][k] * deform_grad[2][1]) -
                 deform_grad[0][1] * (d_deform_grad_dx[1][0][i][k] * deform_grad[2][2] -
                                      d_deform_grad_dx[2][0][i][k] * deform_grad[1][2]) +
                 deform_grad[0][2] * (d_deform_grad_dx[1][0][i][k] * deform_grad[2][1] -
                                      d_deform_grad_dx[2][0][i][k] * deform_grad[1][1]) +
                 deform_grad[0][0] * (deform_grad[1][1] * d_deform_grad_dx[2][2][i][k] -
                                      deform_grad[1][2] * d_deform_grad_dx[2][1][i][k]) -
                 deform_grad[0][1] * (deform_grad[1][0] * d_deform_grad_dx[2][2][i][k] -
                                      deform_grad[2][0] * d_deform_grad_dx[1][2][i][k]) +
                 deform_grad[0][2] * (deform_grad[1][0] * d_deform_grad_dx[2][1][i][k] -
                                      deform_grad[2][0] * d_deform_grad_dx[1][1][i][k]));
            goma_ctx->fem.fv->d_volume_strain_dx[i][k] =
                goma_ctx->fem.fv->d_volume_change_dx[i][k] *
                pow(goma_ctx->fem.fv->volume_change, -2. / 3.);

            goma_ctx->fem.fv->d_volume_change_drs[i][k] =
                (d_deform_grad_drs[0][0][i][k] * (deform_grad[1][1] * deform_grad[2][2] -
                                                  deform_grad[1][2] * deform_grad[2][1]) -
                 d_deform_grad_drs[0][1][i][k] * (deform_grad[1][0] * deform_grad[2][2] -
                                                  deform_grad[2][0] * deform_grad[1][2]) +
                 d_deform_grad_drs[0][2][i][k] * (deform_grad[1][0] * deform_grad[2][1] -
                                                  deform_grad[2][0] * deform_grad[1][1]) +
                 deform_grad[0][0] * (d_deform_grad_drs[1][1][i][k] * deform_grad[2][2] -
                                      d_deform_grad_drs[1][2][i][k] * deform_grad[2][1]) -
                 deform_grad[0][1] * (d_deform_grad_drs[1][0][i][k] * deform_grad[2][2] -
                                      d_deform_grad_drs[2][0][i][k] * deform_grad[1][2]) +
                 deform_grad[0][2] * (d_deform_grad_drs[1][0][i][k] * deform_grad[2][1] -
                                      d_deform_grad_drs[2][0][i][k] * deform_grad[1][1]) +
                 deform_grad[0][0] * (deform_grad[1][1] * d_deform_grad_drs[2][2][i][k] -
                                      deform_grad[1][2] * d_deform_grad_drs[2][1][i][k]) -
                 deform_grad[0][1] * (deform_grad[1][0] * d_deform_grad_drs[2][2][i][k] -
                                      deform_grad[2][0] * d_deform_grad_drs[1][2][i][k]) +
                 deform_grad[0][2] * (deform_grad[1][0] * d_deform_grad_drs[2][1][i][k] -
                                      deform_grad[2][0] * d_deform_grad_drs[1][1][i][k]));
            goma_ctx->fem.fv->d_volume_strain_drs[i][k] =
                goma_ctx->fem.fv->d_volume_change_drs[i][k] *
                pow(goma_ctx->fem.fv->volume_change, -2. / 3.);
          }
        }
      }
      break;
    default:
      GOMA_EH(-1, "Bad dim.");
    }
  } /* end of Non-Linear volume change */

  /*******************************************************************************/
  /* convert Cauchy-Green strain tensor to strain tensor for desired consitutive laws */
  /*******************************************************************************/

  if (goma_ctx->fem.cr->RealSolidFluxModel == LINEAR ||
      goma_ctx->fem.cr->RealSolidFluxModel == NONLINEAR ||
      goma_ctx->fem.cr->RealSolidFluxModel == HOOKEAN_PSTRAIN) {
    for (p = 0; p < dim; p++) {
      for (q = 0; q < dim; q++) {
        goma_ctx->fem.fv->strain[p][q] = cauchy_green[p][q];

        if (goma_ctx->fem.af->Assemble_Jacobian) {
          for (i = 0; i < dim; i++) {
            for (k = 0; k < mdof; k++) {
              goma_ctx->fem.fv->d_strain_dx[p][q][i][k] = d_cauchy_green_dx[p][q][i][k];
            }
            for (k = 0; k < mdof; k++) {
              goma_ctx->fem.fv->d_strain_drs[p][q][i][k] = d_cauchy_green_drs[p][q][i][k];
            }
          }
        }
      }
    }
  } else if (goma_ctx->fem.cr->RealSolidFluxModel == INCOMP_PSTRAIN ||
             goma_ctx->fem.cr->RealSolidFluxModel == INCOMP_3D) {

    /* this is the model of Segalman which removes dilation from the strain tensor */
    factor = 0.5;
    for (p = 0; p < dim; p++) {
      for (q = 0; q < dim; q++) {
        goma_ctx->fem.fv->strain[p][q] =
            factor * delta(p, q) * (1. - pow(goma_ctx->fem.fv->volume_change, 2. / 3.)) +
            cauchy_green[p][q] * pow(goma_ctx->fem.fv->volume_change, 2. / 3.);
      }
    }
    if (goma_ctx->fem.af->Assemble_Jacobian) {
      /*
       * Now find sensitivity with respect to nodal displacements
       */
      for (i = 0; i < dim; i++) {
        for (k = 0; k < mdof; k++) {
          for (p = 0; p < dim; p++) {
            for (q = 0; q < dim; q++) {
              goma_ctx->fem.fv->d_strain_dx[p][q][i][k] =
                  -2. / 3. * factor * delta(p, q) * pow(goma_ctx->fem.fv->volume_change, -1. / 3.) *
                      goma_ctx->fem.fv->d_volume_change_dx[i][k] +
                  (d_cauchy_green_dx[p][q][i][k] * pow(goma_ctx->fem.fv->volume_change, 2. / 3.) +
                   2. / 3. * cauchy_green[p][q] * pow(goma_ctx->fem.fv->volume_change, -1. / 3.) *
                       goma_ctx->fem.fv->d_volume_change_dx[i][k]);
              goma_ctx->fem.fv->d_strain_drs[p][q][i][k] =
                  -2. / 3. * factor * delta(p, q) * pow(goma_ctx->fem.fv->volume_change, -1. / 3.) *
                      goma_ctx->fem.fv->d_volume_change_drs[i][k] +
                  (d_cauchy_green_drs[p][q][i][k] * pow(goma_ctx->fem.fv->volume_change, 2. / 3.) +
                   2. / 3. * cauchy_green[p][q] * pow(goma_ctx->fem.fv->volume_change, -1. / 3.) *
                       goma_ctx->fem.fv->d_volume_change_drs[i][k]);
            }
          }
        }
      }
    }
    /* end of INCOMPRESSIBLE MODELS */
  } else if (goma_ctx->fem.cr->RealSolidFluxModel == INCOMP_PSTRESS ||
             goma_ctx->fem.cr->RealSolidFluxModel == HOOKEAN_PSTRESS) {
    GOMA_EH(GOMA_ERROR, "Need to fix PSTRESS implementation");

  } else
    GOMA_EH(GOMA_ERROR, "Illegal Mesh Constitutive Equation");

  return (status);
}

/******************************************************************************/
/* get_convection_velocity_rs
 * This routine calculates the effective convection velocity to be used
 * in the TOTAL_ALE formulation.  Its counterpart is "get_convection_velocity" in
 * mm_fill_species.c and accounts for all of the lagrangian and arbitrary cases
 * for continuous and porous media. This routine Simply computes the div(rhoVV)
 * term for the real solid where V = Vsfs(FF_T)  If all of this works, then we
 * will stir in the gory details of the porous media here.  For now go back to
 * the old way which is still functional
 */

int get_convection_velocity_rs(
    double vconv[DIM],     /*Calculated convection velocity */
    double vconv_old[DIM], /*Calculated convection velocity at previous time*/
    CONVECTION_VELOCITY_DEPENDENCE_STRUCT *d_vconv,
    double dt,
    double tt) {
  /*
   * Some local variables for convenience...
   */

  int dim, p, q, b, var, i;

  dim = goma_ctx->fem.pd->Num_Dim;

  for (p = 0; p < goma_ctx->problem.vim; p++) {
    vconv[p] = 0.;
  }

  /* initialize sensitivity arrays if necessary */
  if (goma_ctx->fem.af->Assemble_Jacobian && d_vconv != NULL) {
    var = MESH_DISPLACEMENT1;
    if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
      memset(d_vconv->X, 0, DIM * DIM * MDE * sizeof(dbl));
    }

    var = SOLID_DISPLACEMENT1;
    if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
      memset(d_vconv->rs, 0, DIM * DIM * MDE * sizeof(dbl));
    }

    var = VELOCITY1;
    if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
      memset(d_vconv->v, 0, DIM * DIM * MDE * sizeof(dbl));
    }

    var = MASS_FRACTION;
    if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
      memset(d_vconv->C, 0, DIM * MAX_CONC * MDE * sizeof(dbl));
    }

    var = TEMPERATURE;
    if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
      memset(d_vconv->T, 0, DIM * MDE * sizeof(dbl));
    }
  }

  /*
   * Add in convection due to motion of Stress Free State - Pseudo Lagrangian Convection
   *  NOTE: this formulation still assumes mesh is quasi -   static, i.e. that mesh motion
   *        in transient calculations contributes negligibly to the momentum equation
   *    v_solid_lagrangian_convection
   */

  /* First test to see what type of prescribed kinematics model */

  if (goma_ctx->fem.elc_rs->v_mesh_sfs_model > CONSTANT) {
    V_mesh_sfs_model(goma_ctx->fem.elc_rs->u_v_mesh_sfs, goma_ctx->fem.elc_rs->v_mesh_sfs,
                     goma_ctx->fem.elc_rs->v_mesh_sfs_model, -1);
  }

  if (goma_ctx->fem.pd->MeshInertia == 1) {
    if (goma_ctx->fem.pd->TimeIntegration != STEADY)
      GOMA_EH(GOMA_ERROR, "Can't have Unsteady Mesh Inertia ");
    /*
     * Velocity of solid in lab coordinates is the velocity of the stress free state
     * dotted into the deformation gradient tensor for the material
     */
    if (!goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][MESH_DISPLACEMENT1]) {
      for (p = 0; p < dim; p++) {
        vconv[p] += goma_ctx->fem.elc_rs->v_mesh_sfs[p];
      }
    } else {
      for (p = 0; p < dim; p++) {
        for (q = 0; q < dim; q++) {
          vconv[p] += goma_ctx->fem.elc_rs->v_mesh_sfs[q] * goma_ctx->fem.fv->deform_grad_rs[q][p];
        }
      }
      if (goma_ctx->fem.af->Assemble_Jacobian && d_vconv != NULL) {
        for (b = 0; b < dim; b++) {
          var = MESH_DISPLACEMENT1 + b;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; i++) {
              for (p = 0; p < dim; p++) {
                for (q = 0; q < dim; q++) {
                  d_vconv->X[p][b][i] += goma_ctx->fem.elc_rs->v_mesh_sfs[q] *
                                         goma_ctx->fem.fv->d_deform_grad_rs_dx[q][p][b][i];
                }
              }
            }
          }

          var = SOLID_DISPLACEMENT1 + b;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; i++) {
              for (p = 0; p < dim; p++) {
                for (q = 0; q < dim; q++) {
                  d_vconv->rs[p][b][i] += goma_ctx->fem.elc_rs->v_mesh_sfs[q] *
                                          goma_ctx->fem.fv->d_deform_grad_rs_drs[q][p][b][i];
                }
              }
            }
          }
        }
      }
    }
  }

  return 0;
}
/* END of routine get_convection_velocity_rs */

void f_kinematic_displacement_bc(double func[DIM],
                                 double d_func[DIM][MAX_VARIABLE_TYPES + MAX_CONC][MDE],
                                 const int eb_mat,
                                 const int ss_id,
                                 const double *u_pars,
                                 const int len_u)

/******************************************************************************
 *
 *  Function which evaluates the displacment kinematic boundary condition
 *                 n.(d - d_rs)=0
 *
 *            Author: P. R. Schunk    (2/16/98)
 *            Revised:
 *
 ******************************************************************************/

{

  /* Local variables */

  int j, kdir, var, p, mn;
  double phi_j;
  double base_displacement[DIM];
  double base_displacement_rs[DIM];
  double dns[DIM] = {0, 0, 0};
  double origin[DIM] = {0, 0, 0};
  double roll_rad = 0, pt_rad;
  double dir_angle[DIM], t, axis_pt[DIM];
  dbl dns_dX[DIM][DIM];
  dbl dns_drs[DIM][DIM];
  int base_displ_model = TRUE;
/***************************** EXECUTION BEGINS *******************************/
#if 0
  for (p = 0; p<goma_ctx->fem.pd->Num_Dim; p++) base_displacement[p] = goma_ctx->fem.fv->initial_displacements[p];
  for (p = 0; p<goma_ctx->fem.pd->Num_Dim; p++) base_displacement_rs[p] = goma_ctx->fem.fv->initial_displacements[p + DIM];
#else

  if (goma_ctx->fem.af->Assemble_LSA_Mass_Matrix)
    return;
  /*
   * Only want to apply this from a mat that HAS the rs displacement field.
   * In a fluid-solid problem, the fluid may not have such a field defined.
   */
  if (goma_ctx->fem.current_eb_ptr->Elem_Blk_Id == eb_mat) {
    mn = map_mat_index(eb_mat);
    if (goma_ctx->fem.elc_rs_glob[mn]->v_mesh_sfs_model == ROTATIONAL ||
        goma_ctx->fem.elc_rs_glob[mn]->v_mesh_sfs_model == ROTATIONAL_3D) {
      if (len_u < 1)
        GOMA_EH(GOMA_ERROR, "need roll radius parameter on KIN_DISPLACEMENT");
      roll_rad = u_pars[0];
    } else if (goma_ctx->fem.elc_rs_glob[mn]->v_mesh_sfs_model == CONSTANT ||
               goma_ctx->fem.elc_rs_glob[mn]->v_mesh_sfs_model == 0) {
      if (len_u < 4)
        GOMA_WH(-1, "Warning: No plane parameters on KIN_DISPLACEMENT");
    } else {
      GOMA_EH(GOMA_ERROR, "Unknown Conv. Lag. Velocity Model for KIN_DISPL.\n");
    }
    if (goma_ctx->fem.elc_rs_glob[mn]->v_mesh_sfs_model == ROTATIONAL) {
      for (p = 0; p < goma_ctx->fem.pd->Num_Dim; p++)
        origin[p] = goma_ctx->fem.elc_rs_glob[mn]->u_v_mesh_sfs[p + 1];
      pt_rad = 0;
      for (p = 0; p < goma_ctx->fem.pd->Num_Dim; p++) {
        pt_rad += SQUARE(goma_ctx->fem.fv->x[p] - goma_ctx->fem.fv->d_rs[p] - origin[p]);
      }
      pt_rad = sqrt(pt_rad);

      for (p = 0; p < goma_ctx->fem.pd->Num_Dim; p++) {
        dns[p] = origin[p] + roll_rad / pt_rad *
                                 (goma_ctx->fem.fv->x[p] - goma_ctx->fem.fv->d_rs[p] - origin[p]);
        for (int r = 0; r < goma_ctx->fem.pd->Num_Dim; r++) {
          dns_dX[p][r] = -roll_rad *
                         (goma_ctx->fem.fv->x[r] - goma_ctx->fem.fv->d_rs[r] - origin[r]) /
                         (pt_rad * pt_rad * pt_rad) *
                         (goma_ctx->fem.fv->x[p] - goma_ctx->fem.fv->d_rs[p] - origin[p]);
          dns_dX[p][r] += delta(p, r) * roll_rad / pt_rad;
          dns_drs[p][r] = -dns_dX[p][r];
        }
      }
    } else if (goma_ctx->fem.elc_rs_glob[mn]->v_mesh_sfs_model == ROTATIONAL_3D) {
      for (p = 0; p < DIM; p++)
        origin[p] = goma_ctx->fem.elc_rs_glob[mn]->u_v_mesh_sfs[p + 1];
      for (p = 0; p < DIM; p++)
        dir_angle[p] = goma_ctx->fem.elc_rs_glob[mn]->u_v_mesh_sfs[p + 1 + DIM];

      /*  find intersection of axis with normal plane - i.e., locate point on
       *          axis that intersects plane normal to axis that contains local point. */

      for (p = 0; p < goma_ctx->fem.pd->Num_Dim; p++)
        dns[p] = goma_ctx->fem.fv->x[p] - goma_ctx->fem.fv->d_rs[p];
      t = (dir_angle[0] * (dns[0] - origin[0]) + dir_angle[1] * (dns[1] - origin[1]) +
           dir_angle[2] * (dns[2] - origin[2])) /
          (SQUARE(dir_angle[0]) + SQUARE(dir_angle[1]) + SQUARE(dir_angle[2]));
      axis_pt[0] = origin[0] + dir_angle[0] * t;
      axis_pt[1] = origin[1] + dir_angle[1] * t;
      axis_pt[2] = origin[2] + dir_angle[2] * t;

      /*  compute radius and radial direction */

      pt_rad = sqrt(SQUARE(dns[0] - axis_pt[0]) + SQUARE(dns[1] - axis_pt[1]) +
                    SQUARE(dns[2] - axis_pt[2]));
      dns[0] = axis_pt[0] + (dns[0] - axis_pt[0]) * roll_rad / pt_rad;
      dns[1] = axis_pt[1] + (dns[1] - axis_pt[1]) * roll_rad / pt_rad;
      dns[2] = axis_pt[2] + (dns[2] - axis_pt[2]) * roll_rad / pt_rad;

    } else if (len_u >= 4 && (goma_ctx->fem.elc_rs_glob[mn]->v_mesh_sfs_model == CONSTANT ||
                              goma_ctx->fem.elc_rs_glob[mn]->v_mesh_sfs_model == 0)) {
      for (p = 0; p < goma_ctx->fem.pd->Num_Dim; p++)
        dns[p] = goma_ctx->fem.fv->x[p] - goma_ctx->fem.fv->d_rs[p];
      t = -(u_pars[0] * dns[0] + u_pars[1] * dns[1] + u_pars[2] * dns[2] + u_pars[3]) /
          (SQUARE(u_pars[0]) + SQUARE(u_pars[1]) + SQUARE(u_pars[2]));
      for (p = 0; p < goma_ctx->fem.pd->Num_Dim; p++)
        dns[p] += u_pars[p] * t;

      for (p = 0; p < goma_ctx->fem.pd->Num_Dim; p++) {
        for (int r = 0; r < goma_ctx->fem.pd->Num_Dim; r++) {
          dns_dX[p][r] =
              delta(p, r) + -u_pars[p] * u_pars[p] * delta(p, r) /
                                (SQUARE(u_pars[0]) + SQUARE(u_pars[1]) + SQUARE(u_pars[2]));
          dns_drs[p][r] =
              -delta(p, r) + u_pars[p] * u_pars[p] * delta(p, r) /
                                 (SQUARE(u_pars[0]) + SQUARE(u_pars[1]) + SQUARE(u_pars[2]));
        }
      }

    } else {
      GOMA_WH(-1, "Reverting to old base_displacement version.\n");
      base_displ_model = FALSE;
    }

    if (base_displ_model) {
      for (p = 0; p < goma_ctx->fem.pd->Num_Dim; p++)
        base_displacement[p] = goma_ctx->fem.fv->d[p] - goma_ctx->fem.fv->x[p];
      for (p = 0; p < goma_ctx->fem.pd->Num_Dim; p++)
        base_displacement_rs[p] = -dns[p];
    } else {
      for (p = 0; p < goma_ctx->fem.pd->Num_Dim; p++)
        base_displacement[p] = goma_ctx->fem.fv->initial_displacements[p];
      for (p = 0; p < goma_ctx->fem.pd->Num_Dim; p++)
        base_displacement_rs[p] = goma_ctx->fem.fv->initial_displacements[p + DIM];
    }
#endif

    if (goma_ctx->fem.af->Assemble_Jacobian) {
      for (kdir = 0; kdir < goma_ctx->fem.pd->Num_Dim; kdir++) {
        for (p = 0; p < goma_ctx->fem.pd->Num_Dim; p++) {
          var = MESH_DISPLACEMENT1 + p;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
              phi_j = goma_ctx->fem.bf[var]->phi[j];
              d_func[0][var][j] +=
                  ((goma_ctx->fem.fv->d[kdir] - base_displacement[kdir]) -
                   (goma_ctx->fem.fv->d_rs[kdir] - base_displacement_rs[kdir])) *
                      goma_ctx->fem.fv->dsnormal_dx[kdir][p][j] +
                  (delta(kdir, p) - dns_dX[kdir][p]) * phi_j * goma_ctx->fem.fv->snormal[kdir];
            }
          }
        }

        for (p = 0; p < goma_ctx->fem.pd->Num_Dim; p++) {
          var = SOLID_DISPLACEMENT1 + p;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
              phi_j = goma_ctx->fem.bf[var]->phi[j];
              d_func[0][var][j] -=
                  (delta(kdir, p) + dns_drs[kdir][p]) * phi_j * goma_ctx->fem.fv->snormal[kdir];
            }
          }
        }
      }
    } /* end of if Assemble_Jacobian */

    /* Calculate the residual contribution	*/

    for (kdir = 0; kdir < goma_ctx->fem.pd->Num_Dim; kdir++) {
      *func += ((goma_ctx->fem.fv->d[kdir] - base_displacement[kdir]) -
                (goma_ctx->fem.fv->d_rs[kdir] - base_displacement_rs[kdir])) *
               goma_ctx->fem.fv->snormal[kdir];
    }
  }

} /* END of routine f_kinematic_displacement_bc  */

void f_kinematic_displacement_rs_bc(double func[DIM],
                                    double d_func[DIM][MAX_VARIABLE_TYPES + MAX_CONC][MDE],
                                    int eb_mat,
                                    const int id_side,
                                    double xi[DIM],
                                    const Exo_DB *exo)

/******************************************************************************
 *
 *  Function which evaluates the displacment kinematic boundary condition for psuedo solid
 *  and allows leakage according to the shell_delta_h equation
 *                 n.(d - d_rs)=dh
 *
 *            Author: P. R. Schunk    (10/31/2011)
 *            Revised:
 *
 ******************************************************************************/

{

  /* Local variables */

  int j, kdir, var, p;
  double phi_j;
  double base_displacement[DIM];
  /*  double base_displacement_rs[DIM];*/
  int *n_dof = NULL;
  /***************************** EXECUTION BEGINS *******************************/
  for (p = 0; p < goma_ctx->fem.pd->Num_Dim; p++)
    base_displacement[p] = goma_ctx->fem.fv->initial_displacements[p];
  /*  for (p = 0; p<pd->Num_Dim; p++) base_displacement_rs[p] = fv->initial_displacements[p +
   * DIM];*/

  /*
   * Prepare geometry
   */
  int dof_map[MDE];
  n_dof = (int *)array_alloc(1, MAX_VARIABLE_TYPES, sizeof(int));
  lubrication_shell_initialize(n_dof, dof_map, id_side, xi, exo, 1);

  if (goma_ctx->fem.af->Assemble_LSA_Mass_Matrix)
    return;
  /*
   * Only want to apply this from a mat that HAS the rs displacement field.
   * In a fluid-solid problem, the fluid may not have such a field defined.
   */
  if (goma_ctx->fem.current_eb_ptr->Elem_Blk_Id == eb_mat) {
    if (goma_ctx->fem.af->Assemble_Jacobian) {
      for (kdir = 0; kdir < goma_ctx->fem.pd->Num_Dim; kdir++) {
        for (p = 0; p < goma_ctx->fem.pd->Num_Dim; p++) {
          var = MESH_DISPLACEMENT1 + p;
          if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
              phi_j = goma_ctx->fem.bf[var]->phi[j];
              // d_func[0][var][j] +=  (fv->d_rs[kdir] -base_displacement_rs[kdir])
              //	   * fv->dsnormal_dx[kdir][p][j];

              d_func[0][var][j] += ((goma_ctx->fem.fv->d[kdir] - base_displacement[kdir])) *
                                       goma_ctx->fem.fv->dsnormal_dx[kdir][p][j] +
                                   (delta(kdir, p) * phi_j) * goma_ctx->fem.fv->snormal[kdir];
              //		  d_func[0][var][j] +=  ((fv->d[kdir] - base_displacement[kdir]) -
              //					 (fv->d_rs[kdir]
              //-base_displacement_rs[kdir]))
              //		    * fv->dsnormal_dx[kdir][p][j]
              //                                  +(delta(kdir,p)*phi_j)          *
              //                                  fv->snormal[kdir];
            }
          }
        }

        var = SOLID_DISPLACEMENT1 + kdir;
        if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            phi_j = goma_ctx->fem.bf[var]->phi[j];
            //		d_func[0][var][j] -= phi_j * fv->snormal[kdir];
          }
        }
      }

      var = SHELL_DELTAH;
      if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var] != -1) {

        /*** Loop over DOFs (j) ***/
        for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {

          /* Load basis functions (j) */
          phi_j = goma_ctx->fem.bf[var]->phi[j];

          d_func[0][var][j] += phi_j;
        }
      }
    } /* end of if Assemble_Jacobian */

    /* Calculate the residual contribution	*/

    *func += goma_ctx->fem.fv->sh_dh;

    for (kdir = 0; kdir < goma_ctx->fem.pd->Num_Dim; kdir++) {
      //	  *func += ((fv->d[kdir] - base_displacement[kdir]) -
      //		    (fv->d_rs[kdir] -base_displacement_rs[kdir])) * fv->snormal[kdir];
      *func +=
          (goma_ctx->fem.fv->d[kdir] - base_displacement[kdir]) * goma_ctx->fem.fv->snormal[kdir];
    }
  }

} /* END of routine f_kinematic_displacement_rs_bc  */
