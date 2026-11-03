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

/*
 *   add user components to Residual and Jacobian below
 *
 */

if (goma_ctx->fem.af->Assemble_Residual) {
  for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {
    phi_i = goma_ctx->fem.bf[eqn]->phi[i];

    /* Mass term */
    if (pd0->e[0][eqn] & T_MASS) {
      mass = phi_i * (goma_ctx->fem.fv->sh_u - 2 * goma_ctx->fem.fv->qs);
      mass *= pd0->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_MASS)];
      diffusion = 0.0;
      diffusion *= pd0->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_DIFFUSION)];
    }
    res[i] += (mass + diffusion) * wt * h3 * det_J;
    goma_ctx->fem.lec->R[LEC_R_INDEX(peqn, i)] += res[i];
  }
}
/* Include Jacobian contributions from shell variables */
if (goma_ctx->fem.af->Assemble_Jacobian) {
  for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[eqn]; i++) {
    phi_i = goma_ctx->fem.bf[eqn]->phi[i];

    /* J_qs_qs:  Shell sensitivity */
    var = SHELL_USER;
    pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];
    for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
      phi_j = goma_ctx->fem.bf[var]->phi[j];
      phi_t = phi_j;

      /* Mass term only */
      mass = 0.0;
      if (pd0->e[0][eqn] & T_MASS) {
        mass += phi_i * phi_t;
        mass *= pd0->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_MASS)];
      }
      diffusion = 0.0;
      if (pd0->e[0][eqn] & T_MASS) {
        diffusion *= pd0->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_DIFFUSION)];
      }
      jac[i][pvar][j] += (mass + diffusion) * wt * h3 * det_J;
      goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += jac[i][pvar][j];
    }
    /* J_qs_qs:  Shell sensitivity */
    var = SURF_CHARGE;
    pvar = goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][var];
    for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
      phi_j = goma_ctx->fem.bf[var]->phi[j];
      phi_t = phi_j;

      /* Mass term only */
      mass = 0.0;
      if (pd0->e[0][eqn] & T_MASS) {
        mass += phi_i * (-2 * phi_t);
        mass *= pd0->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_MASS)];
      }
      diffusion = 0.0;
      if (pd0->e[0][eqn] & T_MASS) {
        diffusion *= pd0->etm[goma_ctx->fem.pg->imtrx][eqn][(LOG2_DIFFUSION)];
      }
      jac[i][pvar][j] += (mass + diffusion) * wt * h3 * det_J;
      goma_ctx->fem.lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += jac[i][pvar][j];
    }
  }
}
