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
#include <string.h>

#include "bc_colloc.h"
#include "el_geom.h"
#include "load_field_variables.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_fill_ptrs.h"
#include "mm_fill_shell.h"
#include "mm_fill_stress.h"
#include "mm_mp.h"
#include "mm_post_def.h"
#include "rf_bc_const.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "std.h"

/***************************************************************************/
/****************************************************************************/
/****************************************************************************/

void scalar_fv_fill(double **base,
                    double **base_dot,
                    double **base_old,
                    double *phiv,
                    int dofs,
                    double *val,
                    double *val_dot,
                    double *val_old)
/*
 *   scalar_fv_fill:   Kernal operation for calculating the value of a
 *                     scalar field variable at the quadrature point from
 *                     the basis functions.
 *                     Note for the optimal results, this function should
 *                     be inlined during the compile.
 */
{
  int i;
  *val = *val_dot = *val_old = 0.0;
  if (goma_ctx->fem.pd->TimeIntegration == STEADY) {
    for (i = 0; i < dofs; i++) {
      *val += *(base[i]) * phiv[i];
    }
  } else {
    for (i = 0; i < dofs; i++) {
      *val += *(base[i]) * phiv[i];
      *val_dot += *(base_dot[i]) * phiv[i];
      *val_old += *(base_old[i]) * phiv[i];
    }
  }
}

/***************************************************************************/
/****************************************************************************/
/****************************************************************************/

void grad_scalar_fv_fill(double **base, double (*grad_phiv)[DIM], int dofs, double *grad_val)
/*
 */
{
  int i;

  grad_val[0] = grad_val[1] = grad_val[2] = 0.0;
  for (i = 0; i < dofs; i++) {
    grad_val[0] += *(base[i]) * grad_phiv[i][0];
    grad_val[1] += *(base[i]) * grad_phiv[i][1];
    if (VIM == 3)
      grad_val[2] += *(base[i]) * grad_phiv[i][2];
  }
}

void grad_vector_fv_fill(double ***base,
                         double (*grad_phiv)[DIM][DIM][DIM],
                         int dofs,
                         double (*grad_val)[DIM]) {
  int r, i;

  double base_off;
  double (*grad_phiv_off)[DIM];

  memset(grad_val, 0, DIM * DIM * sizeof(double));

  for (r = 0; r < VIM; r++) {
    for (i = 0; i < dofs; i++) {
      base_off = *base[r][i];
      grad_phiv_off = grad_phiv[i][r];

      grad_val[0][0] += base_off * grad_phiv_off[0][0];
      grad_val[1][1] += base_off * grad_phiv_off[1][1];
      grad_val[0][1] += base_off * grad_phiv_off[0][1];
      grad_val[1][0] += base_off * grad_phiv_off[1][0];

      if (VIM == 3) {
        grad_val[2][2] += base_off * grad_phiv_off[2][2];
        grad_val[2][1] += base_off * grad_phiv_off[2][1];
        grad_val[2][0] += base_off * grad_phiv_off[2][0];
        grad_val[1][2] += base_off * grad_phiv_off[1][2];
        grad_val[0][2] += base_off * grad_phiv_off[0][2];
      }

      /*	for ( p=0; p<VIM; p++)
              {
                      for ( q=0; q<VIM; q++)
                      {
                              (grad_val[p][q] += *base[r][i] * grad_phiv[i][r] [p][q];)
                              grad_val[p][q] += base_off * grad_phiv_off[p][q];
                      }
              }*/
    }
  }
}

/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/

int load_fv(void)

/*******************************************************************************
 * load_fv() -- load up values of all relevant field variables at the
 *              current gauss pt
 *
 * input: (assume the appropriate parts of esp, bf, ei, pd are filled in)
 * ------
 *
 *
 * output: ( this routine fills in the following parts of the fv structure)
 * -------
 *		fv->
 *			T -- temperature	(scalar)
 *			v -- velocity		(vector)
 *			d -- mesh displacement	(vector)
 *			c -- concentration	(multiple scalars)
 *		      por -- porous media       (multiple scalars)
 *			P -- pressure		(scalar)
 *			S -- polymer stress	(tensor)
 *			G -- velocity gradient	(tensor)
 *                     pv -- particle velocity  (vector)
 *                     pG -- particle velocity gradient (tensor)
 *              mp->StateVector[] is filled in as well, for
 *                    pertinent entries that make up the specification of
 *                    the state of the material.
 *
 * NOTE: To accommodate shell elements, this function has been modified
 *       so that fv variables are not zeroed out when they are active
 *       on an element block other than the current one.
 *       The check done for variable v is then:
 *          if ( pd->v[pg->imtrx][v] || upd->vp[pg->imtrx][v] == -1 )
 *       In many cases below, this conditional zeroing is done in
 *       a separate small loop before the main one.
 *
 *
 * Return values:
 *		0 -- if things went OK
 *	       -1 -- if a problem occurred
 *
 * Created:	Fri Mar 18 06:44:41 MST 1994 pasacki@sandia.gov
 *
 * Modified:
 ***************************************************************************/
{
  int v;    /* variable type indicator */
  int i;    /* index */
  int p, q; /* dimension indeces */
  int dim;
  int dofs; /* degrees of freedom for a var in the elem */
  int w;    /* concentration species and porous media vars counter */
  int node, index;
  int N;
  int mode;
  int status = 0;
  int v_s[MAX_MODES][DIM][DIM], v_g[DIM][DIM];
  double rho, *stateVector = goma_ctx->fem.mp->StateVector;
  BASIS_FUNCTIONS_STRUCT *bfv;
  int transient_run = goma_ctx->fem.pd->TimeIntegration != STEADY;
  int *pdgv = goma_ctx->fem.pd->gv;

  status = 0;

  /* load eqn and variable number in tensor form */
  if (pdgv[POLYMER_STRESS11]) {
    status = stress_eqn_pointer(v_s);
    GOMA_EH(status, "stress_eqn_pointer(v_s)");
  }
  if (pdgv[VELOCITY_GRADIENT11]) {
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
   * Since it is possible to have a 1D element in a 2D problem,
   * (i.e. shell elements), use the problem dimension instead.
   */
  dim = goma_ctx->fem.pd->Num_Dim;

  /*
   * Temperature...
   *    HKM -> Introduced a function to handle the core scalar fill
   *           operation; it should be inlined for maximum speed.
   *           Added the concept that the reference temperature should
   *           be used, when no temperature variable exists in the solution
   *           vector.
   */

  if (pdgv[TEMPERATURE]) {
    v = TEMPERATURE;
    scalar_fv_fill(goma_ctx->fem.esp->T, goma_ctx->fem.esp_dot->T, goma_ctx->fem.esp_old->T,
                   goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->T), &(goma_ctx->fem.fv_dot->T), &(goma_ctx->fem.fv_old->T));
    stateVector[TEMPERATURE] = goma_ctx->fem.fv->T;
  } /*else if (upd->vp[pg->imtrx][v] == -1) {
      fv->T     = mp->reference[TEMPERATURE];
      fv_old->T = 0.;
      fv_dot->T = 0.;
      } */

  /*
   * Fill...
   */

  if (pdgv[FILL]) {
    v = FILL;
    scalar_fv_fill(goma_ctx->fem.esp->F, goma_ctx->fem.esp_dot->F, goma_ctx->fem.esp_old->F,
                   goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->F), &(goma_ctx->fem.fv_dot->F), &(goma_ctx->fem.fv_old->F));
    stateVector[v] = goma_ctx->fem.fv->F;
    goma_ctx->fem.fv_dot_old->F = 0;
    for (int i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v]; i++) {
      if (goma_ctx->fem.upd->Total_Num_Matrices > 1) {
        //        fv_dot_old->F += *(pg->matrices[upd->matrix_index[v]].xdot_old -
        //        pg->matrices[upd->matrix_index[v]].xdot +
        //                                  esp_dot->F[i]) * bf[v]->phi[i];
      } else {
        goma_ctx->fem.fv_dot_old->F +=
            *(xdot_old_static - xdot_static + goma_ctx->fem.esp_dot->F[i]) *
            goma_ctx->fem.bf[v]->phi[i];
      }
    }
  } /*else if (upd->vp[pg->imtrx][v] == -1) {
      fv->F = fv_old->F = fv_dot->F = 0.;
      } */

  /*
   * Suspension Temperature...
   */

  if (pdgv[BOND_EVOLUTION]) {
    v = BOND_EVOLUTION;
    scalar_fv_fill(
        goma_ctx->fem.esp->nn, goma_ctx->fem.esp_dot->nn, goma_ctx->fem.esp_old->nn,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->nn), &(goma_ctx->fem.fv_dot->nn), &(goma_ctx->fem.fv_old->nn));
    stateVector[v] = goma_ctx->fem.fv->nn;
  } /*else if (upd->vp[pg->imtrx][v] == -1)  {
      fv->nn = fv_old->nn = fv_dot->nn = 0.;
      } */

  /*
   * Voltage...
   */

  if (pdgv[VOLTAGE]) {
    v = VOLTAGE;
    scalar_fv_fill(goma_ctx->fem.esp->V, goma_ctx->fem.esp_dot->V, goma_ctx->fem.esp_old->V,
                   goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->V), &(goma_ctx->fem.fv_dot->V), &(goma_ctx->fem.fv_old->V));
    stateVector[VOLTAGE] = goma_ctx->fem.fv->V;
  } /*else if ( upd->vp[pg->imtrx][v] == -1 ) {
      fv->V = fv_old->V = fv_dot->V = 0.;
      }*/

  /*
   * Surface charge density...
   */

  if (pdgv[SURF_CHARGE]) {
    v = SURF_CHARGE;
    scalar_fv_fill(
        goma_ctx->fem.esp->qs, goma_ctx->fem.esp_dot->qs, goma_ctx->fem.esp_old->qs,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->qs), &(goma_ctx->fem.fv_dot->qs), &(goma_ctx->fem.fv_old->qs));
    stateVector[SURF_CHARGE] = goma_ctx->fem.fv->qs;
  } /*else if ( upd->vp[pg->imtrx][v] == -1 ) {
      fv->qs = fv_old->qs = fv_dot->qs = 0.;
      } */

  /*
   * Structural shell curvature
   */

  if (pdgv[SHELL_CURVATURE]) {
    v = SHELL_CURVATURE;
    scalar_fv_fill(
        goma_ctx->fem.esp->sh_K, goma_ctx->fem.esp_dot->sh_K, goma_ctx->fem.esp_old->sh_K,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->sh_K), &(goma_ctx->fem.fv_dot->sh_K), &(goma_ctx->fem.fv_old->sh_K));
    stateVector[SHELL_CURVATURE] = goma_ctx->fem.fv->sh_K;
  } /*else if ( upd->vp[pg->imtrx][v] == -1 ) {
      fv->sh_K = fv_old->sh_K = fv_dot->sh_K = 0.;
      } */

  if (pdgv[SHELL_CURVATURE2]) {
    v = SHELL_CURVATURE2;
    scalar_fv_fill(
        goma_ctx->fem.esp->sh_K2, goma_ctx->fem.esp_dot->sh_K2, goma_ctx->fem.esp_old->sh_K2,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->sh_K2), &(goma_ctx->fem.fv_dot->sh_K2), &(goma_ctx->fem.fv_old->sh_K2));
    stateVector[SHELL_CURVATURE2] = goma_ctx->fem.fv->sh_K2;
  } /*else if ( upd->vp[pg->imtrx][v] == -1 ) {
      fv->sh_K2 = fv_old->sh_K2 = fv_dot->sh_K2 = 0.;
      } */

  /*
   * Structural shell tension
   */

  if (pdgv[SHELL_TENSION]) {
    v = SHELL_TENSION;
    scalar_fv_fill(goma_ctx->fem.esp->sh_tens, goma_ctx->fem.esp_dot->sh_tens,
                   goma_ctx->fem.esp_old->sh_tens, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->sh_tens), &(goma_ctx->fem.fv_dot->sh_tens),
                   &(goma_ctx->fem.fv_old->sh_tens));
    stateVector[SHELL_TENSION] = goma_ctx->fem.fv->sh_tens;
  } /*else if ( updgvp[v] == -1 ) {
      fv->sh_tens = fv_old->sh_tens = fv_dot->sh_tens = 0.;
      }*/
  /*
   * Structural shell x coordinate
   */

  if (pdgv[SHELL_X]) {
    v = SHELL_X;
    scalar_fv_fill(
        goma_ctx->fem.esp->sh_x, goma_ctx->fem.esp_dot->sh_x, goma_ctx->fem.esp_old->sh_x,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->sh_x), &(goma_ctx->fem.fv_dot->sh_x), &(goma_ctx->fem.fv_old->sh_x));
    stateVector[SHELL_X] = goma_ctx->fem.fv->sh_x;
  } /*else if ( upd->vp[pg->imtrx][v] == -1 ) {
      fv->sh_x = fv_old->sh_x = fv_dot->sh_x    = 0.;
      } */

  /*
   * Structural shell y coordinate
   */

  if (pdgv[SHELL_Y]) {
    v = SHELL_Y;
    scalar_fv_fill(
        goma_ctx->fem.esp->sh_y, goma_ctx->fem.esp_dot->sh_y, goma_ctx->fem.esp_old->sh_y,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->sh_y), &(goma_ctx->fem.fv_dot->sh_y), &(goma_ctx->fem.fv_old->sh_y));
    stateVector[SHELL_Y] = goma_ctx->fem.fv->sh_y;
  } /* else if ( upd->vp[pg->imtrx][v] == -1 ) {
       fv->sh_y = fv_old->sh_y = fv_dot->sh_y    = 0.;
       } */

  /*
   * Shell user
   */

  if (pdgv[SHELL_USER]) {
    v = SHELL_USER;
    scalar_fv_fill(
        goma_ctx->fem.esp->sh_u, goma_ctx->fem.esp_dot->sh_u, goma_ctx->fem.esp_old->sh_u,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->sh_u), &(goma_ctx->fem.fv_dot->sh_u), &(goma_ctx->fem.fv_old->sh_u));
    stateVector[SHELL_USER] = goma_ctx->fem.fv->sh_u;
  } /* else if ( upd->vp[pg->imtrx][v] == -1 ) {
       fv->sh_u = fv_old->sh_u = fv_dot->sh_u    = 0.;
       } */

  /*
   * Shear rate from second invariant of rate of strain tensor
   */

  if (pdgv[SHEAR_RATE]) {
    v = SHEAR_RATE;
    goma_ctx->fem.fv->SH = 0.0;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (i = 0; i < dofs; i++) {
      goma_ctx->fem.fv->SH += *goma_ctx->fem.esp->SH[i] * goma_ctx->fem.bf[v]->phi[i];
    }
  }
  /*  else if ( upd->vp[pg->imtrx][v] == -1)
      {
      fv->SH = 0.0;
      } */

  /* Square of the norm of the potential field, |E|^2 */

  if (pdgv[ENORM]) {
    v = ENORM;
    goma_ctx->fem.fv->Enorm = 0.0;
    goma_ctx->fem.fv_old->Enorm = 0.0;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (i = 0; i < dofs; i++) {
      goma_ctx->fem.fv->Enorm += *goma_ctx->fem.esp->Enorm[i] * goma_ctx->fem.bf[v]->phi[i];
      goma_ctx->fem.fv_old->Enorm += *goma_ctx->fem.esp_old->Enorm[i] * goma_ctx->fem.bf[v]->phi[i];
    }
  } /*  else if ( upd->vp[pg->imtrx][v] == -1) {
        fv->Enorm = 0.0;
        fv_old->Enorm = 0.0;
        } */

  /*
   * Curvature of level set function
   */

  if (pdgv[CURVATURE]) {
    v = CURVATURE;
    scalar_fv_fill(goma_ctx->fem.esp->H, goma_ctx->fem.esp_dot->H, goma_ctx->fem.esp_old->H,
                   goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->H), &(goma_ctx->fem.fv_dot->H), &(goma_ctx->fem.fv_old->H));
  }
  /*  else if ( upd->vp[pg->imtrx][v] == -1)
      {
      fv->H = fv_dot->H = fv_old->H = 0.0;
      }
  */

  /*
   *  Normal to level set function
   */

  if (pdgv[NORMAL1]) {
    v = NORMAL1;
    scalar_fv_fill(
        goma_ctx->fem.esp->n[0], goma_ctx->fem.esp_dot->n[0], goma_ctx->fem.esp_old->n[0],
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->n[0]), &(goma_ctx->fem.fv_dot->n[0]), &(goma_ctx->fem.fv_old->n[0]));
  }
  /*  else if ( upd->vp[pg->imtrx][v] == -1)
      {
      fv->n[0] = fv_dot->n[0] = fv_old->n[0] = 0.0;
      } */

  if (pdgv[NORMAL2]) {
    v = NORMAL2;
    scalar_fv_fill(
        goma_ctx->fem.esp->n[1], goma_ctx->fem.esp_dot->n[1], goma_ctx->fem.esp_old->n[1],
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->n[1]), &(goma_ctx->fem.fv_dot->n[1]), &(goma_ctx->fem.fv_old->n[1]));
  }
  /* else if ( upd->vp[pg->imtrx][v] == -1)
     {
     fv->n[1] = fv_dot->n[1] = fv_old->n[1] = 0.0;
     }*/

  if (pdgv[NORMAL3]) {
    v = NORMAL3;
    scalar_fv_fill(
        goma_ctx->fem.esp->n[2], goma_ctx->fem.esp_dot->n[2], goma_ctx->fem.esp_old->n[2],
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->n[2]), &(goma_ctx->fem.fv_dot->n[2]), &(goma_ctx->fem.fv_old->n[2]));
  }
  /*  else if ( upd->vp[pg->imtrx][v] == -1)
      {
      fv->n[2] = fv_dot->n[2] = fv_old->n[2] = 0.0;
      } */

  /*
   *  shell element orientation angles
   */

  if (pdgv[SHELL_ANGLE1]) {
    v = SHELL_ANGLE1;
    goma_ctx->fem.fv->sh_ang[0] = 0.;
    for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v]; i++) {
      if ((*goma_ctx->fem.esp->sh_ang[0][i] - *goma_ctx->fem.esp->sh_ang[0][0]) > M_PIE)
        goma_ctx->fem.fv->sh_ang[0] +=
            goma_ctx->fem.bf[v]->phi[i] * (*goma_ctx->fem.esp->sh_ang[0][i] - 2. * M_PIE);
      else if ((*goma_ctx->fem.esp->sh_ang[0][i] - *goma_ctx->fem.esp->sh_ang[0][0]) < -M_PIE)
        goma_ctx->fem.fv->sh_ang[0] +=
            goma_ctx->fem.bf[v]->phi[i] * (*goma_ctx->fem.esp->sh_ang[0][i] + 2. * M_PIE);
      else
        goma_ctx->fem.fv->sh_ang[0] +=
            goma_ctx->fem.bf[v]->phi[i] * *goma_ctx->fem.esp->sh_ang[0][i];
    }
    /* surely no one will be using these */
    goma_ctx->fem.fv_dot->sh_ang[0] = goma_ctx->fem.fv_old->sh_ang[0] = 0.0;
  } /*
      else if ( upd->vp[pg->imtrx][v] == -1)
      {
      fv->sh_ang[0] = fv_dot->sh_ang[0] = fv_old->sh_ang[0] = 0.0;
      }*/

  if (pdgv[SHELL_ANGLE2]) {
    v = SHELL_ANGLE2;
    goma_ctx->fem.fv->sh_ang[1] = 0.;
    for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v]; i++) {
      if ((*goma_ctx->fem.esp->sh_ang[1][i] - *goma_ctx->fem.esp->sh_ang[1][0]) > M_PIE)
        goma_ctx->fem.fv->sh_ang[1] +=
            goma_ctx->fem.bf[v]->phi[i] * (*goma_ctx->fem.esp->sh_ang[1][i] - 2. * M_PIE);
      else if ((*goma_ctx->fem.esp->sh_ang[1][i] - *goma_ctx->fem.esp->sh_ang[1][0]) < -M_PIE)
        goma_ctx->fem.fv->sh_ang[1] +=
            goma_ctx->fem.bf[v]->phi[i] * (*goma_ctx->fem.esp->sh_ang[1][i] + 2. * M_PIE);
      else
        goma_ctx->fem.fv->sh_ang[1] +=
            goma_ctx->fem.bf[v]->phi[i] * *goma_ctx->fem.esp->sh_ang[1][i];
    }
    /* surely no one will be using these */
    goma_ctx->fem.fv_dot->sh_ang[1] = goma_ctx->fem.fv_old->sh_ang[1] = 0.0;
  }
  /*  else if ( upd->vp[pg->imtrx][v] == -1)
      {
      fv->sh_ang[1] = fv_dot->sh_ang[1] = fv_old->sh_ang[1] = 0.0;
      }*/

  /*
   * Surface Rheo shell piece
   */

  if (pdgv[SHELL_SURF_DIV_V]) {
    v = SHELL_SURF_DIV_V;
    scalar_fv_fill(goma_ctx->fem.esp->div_s_v, goma_ctx->fem.esp_dot->div_s_v,
                   goma_ctx->fem.esp_old->div_s_v, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->div_s_v), &(goma_ctx->fem.fv_dot->div_s_v),
                   &(goma_ctx->fem.fv_old->div_s_v));
    stateVector[SHELL_SURF_DIV_V] = goma_ctx->fem.fv->div_s_v;
  } /*else if ( upd->vp[pg->imtrx][v] == -1 ) {
      fv->div_s_v = fv_old->div_s_v = fv_dot->div_s_v    = 0.;
      }*/

  if (pdgv[SHELL_SURF_CURV]) {
    v = SHELL_SURF_CURV;
    scalar_fv_fill(
        goma_ctx->fem.esp->curv, goma_ctx->fem.esp_dot->curv, goma_ctx->fem.esp_old->curv,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->curv), &(goma_ctx->fem.fv_dot->curv), &(goma_ctx->fem.fv_old->curv));
    stateVector[SHELL_SURF_CURV] = goma_ctx->fem.fv->curv;
  } /*else if ( upd->vp[pg->imtrx][v] == -1 ) {
      fv->curv = fv_old->curv = fv_dot->curv    = 0.;
      }*/

  if (pdgv[N_DOT_CURL_V]) {
    v = N_DOT_CURL_V;
    scalar_fv_fill(goma_ctx->fem.esp->n_dot_curl_s_v, goma_ctx->fem.esp_dot->n_dot_curl_s_v,
                   goma_ctx->fem.esp_old->n_dot_curl_s_v, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->n_dot_curl_s_v), &(goma_ctx->fem.fv_dot->n_dot_curl_s_v),
                   &(goma_ctx->fem.fv_old->n_dot_curl_s_v));
    stateVector[N_DOT_CURL_V] = goma_ctx->fem.fv->n_dot_curl_s_v;
  } /*else if ( upd->vp[pg->imtrx][v] == -1 ) {
      fv->n_dot_curl_s_v = fv_old->n_dot_curl_s_v = fv_dot->n_dot_curl_s_v    = 0.;
      }*/

  if (pdgv[GRAD_S_V_DOT_N1]) {
    v = GRAD_S_V_DOT_N1;
    scalar_fv_fill(goma_ctx->fem.esp->grad_v_dot_n[0], goma_ctx->fem.esp_dot->grad_v_dot_n[0],
                   goma_ctx->fem.esp_old->grad_v_dot_n[0], goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->grad_v_dot_n[0]), &(goma_ctx->fem.fv_dot->grad_v_dot_n[0]),
                   &(goma_ctx->fem.fv_old->grad_v_dot_n[0]));
    stateVector[GRAD_S_V_DOT_N1] = goma_ctx->fem.fv->grad_v_dot_n[0];
  } /* else if ( upd->vp[pg->imtrx][v] == -1 ) {
       fv->grad_v_dot_n[0] = fv_old->grad_v_dot_n[0] = fv_dot->grad_v_dot_n[0]    = 0.;
       }*/

  if (pdgv[GRAD_S_V_DOT_N2]) {
    v = GRAD_S_V_DOT_N2;
    scalar_fv_fill(goma_ctx->fem.esp->grad_v_dot_n[1], goma_ctx->fem.esp_dot->grad_v_dot_n[1],
                   goma_ctx->fem.esp_old->grad_v_dot_n[1], goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->grad_v_dot_n[1]), &(goma_ctx->fem.fv_dot->grad_v_dot_n[1]),
                   &(goma_ctx->fem.fv_old->grad_v_dot_n[1]));
    stateVector[GRAD_S_V_DOT_N2] = goma_ctx->fem.fv->grad_v_dot_n[1];
  } /* else if ( upd->vp[pg->imtrx][v] == -1 ) {
       fv->grad_v_dot_n[1] = fv_old->grad_v_dot_n[1] = fv_dot->grad_v_dot_n[1]    = 0.;


       } */

  if (pdgv[GRAD_S_V_DOT_N3]) {
    v = GRAD_S_V_DOT_N3;
    scalar_fv_fill(goma_ctx->fem.esp->grad_v_dot_n[2], goma_ctx->fem.esp_dot->grad_v_dot_n[2],
                   goma_ctx->fem.esp_old->grad_v_dot_n[2], goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->grad_v_dot_n[2]), &(goma_ctx->fem.fv_dot->grad_v_dot_n[2]),
                   &(goma_ctx->fem.fv_old->grad_v_dot_n[2]));
    stateVector[GRAD_S_V_DOT_N3] = goma_ctx->fem.fv->grad_v_dot_n[2];
  } /*else if ( upd->vp[pg->imtrx][v] == -1 ) {
      fv->grad_v_dot_n[2] = fv_old->grad_v_dot_n[2] = fv_dot->grad_v_dot_n[2]    = 0.;
      }*/

  if (pdgv[SHELL_DIFF_FLUX]) {
    v = SHELL_DIFF_FLUX;
    scalar_fv_fill(
        goma_ctx->fem.esp->sh_J, goma_ctx->fem.esp_dot->sh_J, goma_ctx->fem.esp_old->sh_J,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->sh_J), &(goma_ctx->fem.fv_dot->sh_J), &(goma_ctx->fem.fv_old->sh_J));
    stateVector[SHELL_DIFF_FLUX] = goma_ctx->fem.fv->sh_J;
  } /*else if ( upd->vp[pg->imtrx][v] == -1 ) {
      fv->sh_J = fv_old->sh_J = fv_dot->sh_J = 0.;
      }*/

  if (pdgv[SHELL_DIFF_CURVATURE]) {
    v = SHELL_DIFF_CURVATURE;
    scalar_fv_fill(
        goma_ctx->fem.esp->sh_Kd, goma_ctx->fem.esp_dot->sh_Kd, goma_ctx->fem.esp_old->sh_Kd,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->sh_Kd), &(goma_ctx->fem.fv_dot->sh_Kd), &(goma_ctx->fem.fv_old->sh_Kd));
    stateVector[SHELL_DIFF_CURVATURE] = goma_ctx->fem.fv->sh_Kd;
  } /* else if ( upd->vp[pg->imtrx][v] == -1 ) {
       fv->sh_Kd = fv_old->sh_Kd = fv_dot->sh_Kd = 0.;
       }*/

  if (pdgv[SHELL_NORMAL1]) {
    v = SHELL_NORMAL1;
    scalar_fv_fill(
        goma_ctx->fem.esp->n[0], goma_ctx->fem.esp_dot->n[0], goma_ctx->fem.esp_old->n[0],
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->n[0]), &(goma_ctx->fem.fv_dot->n[0]), &(goma_ctx->fem.fv_old->n[0]));
    stateVector[SHELL_NORMAL1] = goma_ctx->fem.fv->n[0];
  } /*else if ( upd->vp[pg->imtrx][v] == -1 ) {
      fv->n[0] = fv_old->n[0] = fv_dot->n[0] = 0.;
      }*/

  if (pdgv[SHELL_NORMAL2]) {
    v = SHELL_NORMAL2;
    scalar_fv_fill(
        goma_ctx->fem.esp->n[1], goma_ctx->fem.esp_dot->n[1], goma_ctx->fem.esp_old->n[1],
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->n[1]), &(goma_ctx->fem.fv_dot->n[1]), &(goma_ctx->fem.fv_old->n[1]));
    stateVector[SHELL_NORMAL2] = goma_ctx->fem.fv->n[1];
  } /*else if ( upd->vp[pg->imtrx][v] == -1 ) {
      fv->n[1] = fv_old->n[1] = fv_dot->n[1] = 0.;
      }*/

  if (pdgv[SHELL_NORMAL3]) {
    v = SHELL_NORMAL3;
    scalar_fv_fill(
        goma_ctx->fem.esp->n[2], goma_ctx->fem.esp_dot->n[2], goma_ctx->fem.esp_old->n[2],
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->n[2]), &(goma_ctx->fem.fv_dot->n[2]), &(goma_ctx->fem.fv_old->n[2]));
    stateVector[SHELL_NORMAL3] = goma_ctx->fem.fv->n[2];
  } /*else if ( upd->vp[pg->imtrx][v] == -1 ) {
      fv->n[2] = fv_old->n[2] = fv_dot->n[2] = 0.;
      }*/

  if (pdgv[SHELL_NORMAL1] && pdgv[SHELL_NORMAL2] && pdgv[SHELL_NORMAL3]) {
    memset(goma_ctx->fem.fv->d_n_dxi, 0.0, sizeof(double) * DIM * DIM);
    for (i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[SHELL_NORMAL1]; i++) {
      goma_ctx->fem.fv->d_n_dxi[0][0] +=
          *goma_ctx->fem.esp->n[0][i] * goma_ctx->fem.bf[SHELL_NORMAL1]->dphidxi[i][0];
      goma_ctx->fem.fv->d_n_dxi[1][0] +=
          *goma_ctx->fem.esp->n[1][i] * goma_ctx->fem.bf[SHELL_NORMAL2]->dphidxi[i][0];
      goma_ctx->fem.fv->d_n_dxi[2][0] +=
          *goma_ctx->fem.esp->n[2][i] * goma_ctx->fem.bf[SHELL_NORMAL3]->dphidxi[i][0];

      goma_ctx->fem.fv->d_n_dxi[0][1] +=
          *goma_ctx->fem.esp->n[0][i] * goma_ctx->fem.bf[SHELL_NORMAL1]->dphidxi[i][1];
      goma_ctx->fem.fv->d_n_dxi[1][1] +=
          *goma_ctx->fem.esp->n[1][i] * goma_ctx->fem.bf[SHELL_NORMAL2]->dphidxi[i][1];
      goma_ctx->fem.fv->d_n_dxi[2][1] +=
          *goma_ctx->fem.esp->n[2][i] * goma_ctx->fem.bf[SHELL_NORMAL3]->dphidxi[i][1];
    }
  }

  /*
   *	Acoustic Pressure
   */

  if (pdgv[ACOUS_PREAL]) {
    v = ACOUS_PREAL;
    scalar_fv_fill(
        goma_ctx->fem.esp->apr, goma_ctx->fem.esp_dot->apr, goma_ctx->fem.esp_old->apr,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->apr), &(goma_ctx->fem.fv_dot->apr), &(goma_ctx->fem.fv_old->apr));
    stateVector[ACOUS_PREAL] = goma_ctx->fem.fv->apr;
  } /*else if (upd->vp[pg->imtrx][v] == -1) {
      fv->apr = fv_old->apr = fv_dot->apr = 0.;
      }*/

  if (pdgv[ACOUS_PIMAG]) {
    v = ACOUS_PIMAG;
    scalar_fv_fill(
        goma_ctx->fem.esp->api, goma_ctx->fem.esp_dot->api, goma_ctx->fem.esp_old->api,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->api), &(goma_ctx->fem.fv_dot->api), &(goma_ctx->fem.fv_old->api));
    stateVector[ACOUS_PIMAG] = goma_ctx->fem.fv->api;
  } /*else if (upd->vp[pg->imtrx][v] == -1) {
      fv->api = fv_old->api = fv_dot->api = 0.;
      }*/

  if (pdgv[ACOUS_REYN_STRESS]) {
    v = ACOUS_REYN_STRESS;
    scalar_fv_fill(
        goma_ctx->fem.esp->ars, goma_ctx->fem.esp_dot->ars, goma_ctx->fem.esp_old->ars,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->ars), &(goma_ctx->fem.fv_dot->ars), &(goma_ctx->fem.fv_old->ars));
    stateVector[ACOUS_REYN_STRESS] = goma_ctx->fem.fv->ars;
  } /*else if (upd->vp[pg->imtrx][v] == -1) {
      fv->ars = fv_old->ars = fv_dot->ars = 0.;
      }*/

  if (pdgv[SHELL_BDYVELO]) {
    v = SHELL_BDYVELO;
    scalar_fv_fill(
        goma_ctx->fem.esp->sh_bv, goma_ctx->fem.esp_dot->sh_bv, goma_ctx->fem.esp_old->sh_bv,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->sh_bv), &(goma_ctx->fem.fv_dot->sh_bv), &(goma_ctx->fem.fv_old->sh_bv));
    stateVector[SHELL_BDYVELO] = goma_ctx->fem.fv->sh_bv;
  }

  if (pdgv[SHELL_LUBP]) {
    v = SHELL_LUBP;
    scalar_fv_fill(
        goma_ctx->fem.esp->sh_p, goma_ctx->fem.esp_dot->sh_p, goma_ctx->fem.esp_old->sh_p,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->sh_p), &(goma_ctx->fem.fv_dot->sh_p), &(goma_ctx->fem.fv_old->sh_p));
    stateVector[SHELL_LUBP] = goma_ctx->fem.fv->sh_p;
  }

  if (pdgv[LUBP]) {
    v = LUBP;
    scalar_fv_fill(
        goma_ctx->fem.esp->lubp, goma_ctx->fem.esp_dot->lubp, goma_ctx->fem.esp_old->lubp,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->lubp), &(goma_ctx->fem.fv_dot->lubp), &(goma_ctx->fem.fv_old->lubp));
    stateVector[LUBP] = goma_ctx->fem.fv->lubp;
  }

  if (pdgv[LUBP_2]) {
    v = LUBP_2;
    scalar_fv_fill(goma_ctx->fem.esp->lubp_2, goma_ctx->fem.esp_dot->lubp_2,
                   goma_ctx->fem.esp_old->lubp_2, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->lubp_2), &(goma_ctx->fem.fv_dot->lubp_2),
                   &(goma_ctx->fem.fv_old->lubp_2));
    stateVector[LUBP_2] = goma_ctx->fem.fv->lubp_2;
  }

  if (pdgv[SHELL_FILMP]) {
    v = SHELL_FILMP;
    scalar_fv_fill(
        goma_ctx->fem.esp->sh_fp, goma_ctx->fem.esp_dot->sh_fp, goma_ctx->fem.esp_old->sh_fp,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->sh_fp), &(goma_ctx->fem.fv_dot->sh_fp), &(goma_ctx->fem.fv_old->sh_fp));
    stateVector[SHELL_FILMP] = goma_ctx->fem.fv->sh_fp;
  }

  if (pdgv[SHELL_FILMH]) {
    v = SHELL_FILMH;
    scalar_fv_fill(
        goma_ctx->fem.esp->sh_fh, goma_ctx->fem.esp_dot->sh_fh, goma_ctx->fem.esp_old->sh_fh,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->sh_fh), &(goma_ctx->fem.fv_dot->sh_fh), &(goma_ctx->fem.fv_old->sh_fh));
    stateVector[SHELL_FILMH] = goma_ctx->fem.fv->sh_fh;
  }

  if (pdgv[SHELL_PARTC]) {
    v = SHELL_PARTC;
    scalar_fv_fill(
        goma_ctx->fem.esp->sh_pc, goma_ctx->fem.esp_dot->sh_pc, goma_ctx->fem.esp_old->sh_pc,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->sh_pc), &(goma_ctx->fem.fv_dot->sh_pc), &(goma_ctx->fem.fv_old->sh_pc));
    stateVector[SHELL_PARTC] = goma_ctx->fem.fv->sh_pc;
  }

  if (pdgv[SHELL_SAT_CLOSED]) {
    v = SHELL_SAT_CLOSED;
    scalar_fv_fill(goma_ctx->fem.esp->sh_sat_closed, goma_ctx->fem.esp_dot->sh_sat_closed,
                   goma_ctx->fem.esp_old->sh_sat_closed, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->sh_sat_closed), &(goma_ctx->fem.fv_dot->sh_sat_closed),
                   &(goma_ctx->fem.fv_old->sh_sat_closed));
    stateVector[SHELL_SAT_CLOSED] = goma_ctx->fem.fv->sh_sat_closed;
  }
  if (pdgv[SHELL_PRESS_OPEN]) {
    v = SHELL_PRESS_OPEN;
    scalar_fv_fill(goma_ctx->fem.esp->sh_p_open, goma_ctx->fem.esp_dot->sh_p_open,
                   goma_ctx->fem.esp_old->sh_p_open, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->sh_p_open), &(goma_ctx->fem.fv_dot->sh_p_open),
                   &(goma_ctx->fem.fv_old->sh_p_open));
    stateVector[SHELL_PRESS_OPEN] = goma_ctx->fem.fv->sh_p_open;
  }
  if (pdgv[SHELL_PRESS_OPEN_2]) {
    v = SHELL_PRESS_OPEN_2;
    scalar_fv_fill(goma_ctx->fem.esp->sh_p_open_2, goma_ctx->fem.esp_dot->sh_p_open_2,
                   goma_ctx->fem.esp_old->sh_p_open_2, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->sh_p_open_2), &(goma_ctx->fem.fv_dot->sh_p_open_2),
                   &(goma_ctx->fem.fv_old->sh_p_open_2));
    stateVector[SHELL_PRESS_OPEN_2] = goma_ctx->fem.fv->sh_p_open_2;
  }
  if (pdgv[SHELL_SAT_1]) {
    v = SHELL_SAT_1;
    scalar_fv_fill(goma_ctx->fem.esp->sh_sat_1, goma_ctx->fem.esp_dot->sh_sat_1,
                   goma_ctx->fem.esp_old->sh_sat_1, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v], &(goma_ctx->fem.fv->sh_sat_1),
                   &(goma_ctx->fem.fv_dot->sh_sat_1), &(goma_ctx->fem.fv_old->sh_sat_1));
    stateVector[SHELL_SAT_1] = goma_ctx->fem.fv->sh_sat_1;
  }
  if (pdgv[SHELL_SAT_2]) {
    v = SHELL_SAT_2;
    scalar_fv_fill(goma_ctx->fem.esp->sh_sat_2, goma_ctx->fem.esp_dot->sh_sat_2,
                   goma_ctx->fem.esp_old->sh_sat_2, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v], &(goma_ctx->fem.fv->sh_sat_2),
                   &(goma_ctx->fem.fv_dot->sh_sat_2), &(goma_ctx->fem.fv_old->sh_sat_2));
    stateVector[SHELL_SAT_2] = goma_ctx->fem.fv->sh_sat_2;
  }
  if (pdgv[SHELL_SAT_3]) {
    v = SHELL_SAT_3;
    scalar_fv_fill(goma_ctx->fem.esp->sh_sat_3, goma_ctx->fem.esp_dot->sh_sat_3,
                   goma_ctx->fem.esp_old->sh_sat_3, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v], &(goma_ctx->fem.fv->sh_sat_3),
                   &(goma_ctx->fem.fv_dot->sh_sat_3), &(goma_ctx->fem.fv_old->sh_sat_3));
    stateVector[SHELL_SAT_3] = goma_ctx->fem.fv->sh_sat_3;
  }
  if (pdgv[SHELL_TEMPERATURE]) {
    v = SHELL_TEMPERATURE;
    scalar_fv_fill(
        goma_ctx->fem.esp->sh_t, goma_ctx->fem.esp_dot->sh_t, goma_ctx->fem.esp_old->sh_t,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->sh_t), &(goma_ctx->fem.fv_dot->sh_t), &(goma_ctx->fem.fv_old->sh_t));
    stateVector[SHELL_TEMPERATURE] = goma_ctx->fem.fv->sh_t;
  }
  if (pdgv[SHELL_DELTAH]) {
    v = SHELL_DELTAH;
    scalar_fv_fill(
        goma_ctx->fem.esp->sh_dh, goma_ctx->fem.esp_dot->sh_dh, goma_ctx->fem.esp_old->sh_dh,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->sh_dh), &(goma_ctx->fem.fv_dot->sh_dh), &(goma_ctx->fem.fv_old->sh_dh));
    stateVector[SHELL_DELTAH] = goma_ctx->fem.fv->sh_dh;
  }
  if (pdgv[SHELL_LUB_CURV]) {
    v = SHELL_LUB_CURV;
    scalar_fv_fill(goma_ctx->fem.esp->sh_l_curv, goma_ctx->fem.esp_dot->sh_l_curv,
                   goma_ctx->fem.esp_old->sh_l_curv, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->sh_l_curv), &(goma_ctx->fem.fv_dot->sh_l_curv),
                   &(goma_ctx->fem.fv_old->sh_l_curv));
    stateVector[SHELL_LUB_CURV] = goma_ctx->fem.fv->sh_l_curv;
  }
  if (pdgv[SHELL_LUB_CURV_2]) {
    v = SHELL_LUB_CURV_2;
    scalar_fv_fill(goma_ctx->fem.esp->sh_l_curv_2, goma_ctx->fem.esp_dot->sh_l_curv_2,
                   goma_ctx->fem.esp_old->sh_l_curv_2, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->sh_l_curv_2), &(goma_ctx->fem.fv_dot->sh_l_curv_2),
                   &(goma_ctx->fem.fv_old->sh_l_curv_2));
    stateVector[SHELL_LUB_CURV] = goma_ctx->fem.fv->sh_l_curv;
  }
  if (pdgv[SHELL_SAT_GASN]) {
    v = SHELL_SAT_GASN;
    scalar_fv_fill(goma_ctx->fem.esp->sh_sat_gasn, goma_ctx->fem.esp_dot->sh_sat_gasn,
                   goma_ctx->fem.esp_old->sh_sat_gasn, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->sh_sat_gasn), &(goma_ctx->fem.fv_dot->sh_sat_gasn),
                   &(goma_ctx->fem.fv_old->sh_sat_gasn));
    stateVector[SHELL_SAT_GASN] = goma_ctx->fem.fv->sh_sat_gasn;
  }
  if (pdgv[SHELL_SHEAR_TOP]) {
    v = SHELL_SHEAR_TOP;
    scalar_fv_fill(goma_ctx->fem.esp->sh_shear_top, goma_ctx->fem.esp_dot->sh_shear_top,
                   goma_ctx->fem.esp_old->sh_shear_top, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->sh_shear_top), &(goma_ctx->fem.fv_dot->sh_shear_top),
                   &(goma_ctx->fem.fv_old->sh_shear_top));
    stateVector[SHELL_SHEAR_TOP] = goma_ctx->fem.fv->sh_shear_top;
  }
  if (pdgv[SHELL_SHEAR_BOT]) {
    v = SHELL_SHEAR_BOT;
    scalar_fv_fill(goma_ctx->fem.esp->sh_shear_bot, goma_ctx->fem.esp_dot->sh_shear_bot,
                   goma_ctx->fem.esp_old->sh_shear_bot, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->sh_shear_bot), &(goma_ctx->fem.fv_dot->sh_shear_bot),
                   &(goma_ctx->fem.fv_old->sh_shear_bot));
    stateVector[SHELL_SHEAR_BOT] = goma_ctx->fem.fv->sh_shear_bot;
  }
  if (pdgv[SHELL_CROSS_SHEAR]) {
    v = SHELL_CROSS_SHEAR;
    scalar_fv_fill(goma_ctx->fem.esp->sh_cross_shear, goma_ctx->fem.esp_dot->sh_cross_shear,
                   goma_ctx->fem.esp_old->sh_cross_shear, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->sh_cross_shear), &(goma_ctx->fem.fv_dot->sh_cross_shear),
                   &(goma_ctx->fem.fv_old->sh_cross_shear));
    stateVector[SHELL_CROSS_SHEAR] = goma_ctx->fem.fv->sh_cross_shear;
  }
  if (pdgv[MAX_STRAIN]) {
    v = MAX_STRAIN;
    scalar_fv_fill(goma_ctx->fem.esp->max_strain, goma_ctx->fem.esp_dot->max_strain,
                   goma_ctx->fem.esp_old->max_strain, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->max_strain), &(goma_ctx->fem.fv_dot->max_strain),
                   &(goma_ctx->fem.fv_old->max_strain));
    stateVector[MAX_STRAIN] = goma_ctx->fem.fv->max_strain;
  }
  if (pdgv[CUR_STRAIN]) {
    v = CUR_STRAIN;
    scalar_fv_fill(goma_ctx->fem.esp->cur_strain, goma_ctx->fem.esp_dot->cur_strain,
                   goma_ctx->fem.esp_old->cur_strain, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->cur_strain), &(goma_ctx->fem.fv_dot->cur_strain),
                   &(goma_ctx->fem.fv_old->cur_strain));
    stateVector[CUR_STRAIN] = goma_ctx->fem.fv->cur_strain;
  }

  if (pdgv[EDDY_NU]) {
    v = EDDY_NU;
    scalar_fv_fill(goma_ctx->fem.esp->eddy_nu, goma_ctx->fem.esp_dot->eddy_nu,
                   goma_ctx->fem.esp_old->eddy_nu, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->eddy_nu), &(goma_ctx->fem.fv_dot->eddy_nu),
                   &(goma_ctx->fem.fv_old->eddy_nu));
    stateVector[EDDY_NU] = goma_ctx->fem.fv->eddy_nu;
  }

  if (pdgv[TURB_K]) {
    v = TURB_K;
    scalar_fv_fill(goma_ctx->fem.esp->turb_k, goma_ctx->fem.esp_dot->turb_k,
                   goma_ctx->fem.esp_old->turb_k, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->turb_k), &(goma_ctx->fem.fv_dot->turb_k),
                   &(goma_ctx->fem.fv_old->turb_k));
    stateVector[TURB_K] = goma_ctx->fem.fv->turb_k;
  }

  if (pdgv[TURB_OMEGA]) {
    v = TURB_OMEGA;
    scalar_fv_fill(goma_ctx->fem.esp->turb_omega, goma_ctx->fem.esp_dot->turb_omega,
                   goma_ctx->fem.esp_old->turb_omega, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->turb_omega), &(goma_ctx->fem.fv_dot->turb_omega),
                   &(goma_ctx->fem.fv_old->turb_omega));
    stateVector[TURB_OMEGA] = goma_ctx->fem.fv->turb_omega;
  }

  if (pdgv[FILM_HEIGHT]) {
    v = FILM_HEIGHT;
    scalar_fv_fill(goma_ctx->fem.esp->film_height, goma_ctx->fem.esp_dot->film_height,
                   goma_ctx->fem.esp_old->film_height, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->film_height), &(goma_ctx->fem.fv_dot->film_height),
                   &(goma_ctx->fem.fv_old->film_height));
    stateVector[FILM_HEIGHT] = goma_ctx->fem.fv->film_height;
  }

  if (pdgv[TURB_K]) {
    v = TURB_K;
    scalar_fv_fill(goma_ctx->fem.esp->turb_k, goma_ctx->fem.esp_dot->turb_k,
                   goma_ctx->fem.esp_old->turb_k, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->turb_k), &(goma_ctx->fem.fv_dot->turb_k),
                   &(goma_ctx->fem.fv_old->turb_k));
    stateVector[TURB_K] = goma_ctx->fem.fv->turb_k;
  }

  if (pdgv[TURB_OMEGA]) {
    v = TURB_OMEGA;
    scalar_fv_fill(goma_ctx->fem.esp->turb_omega, goma_ctx->fem.esp_dot->turb_omega,
                   goma_ctx->fem.esp_old->turb_omega, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->turb_omega), &(goma_ctx->fem.fv_dot->turb_omega),
                   &(goma_ctx->fem.fv_old->turb_omega));
    stateVector[TURB_OMEGA] = goma_ctx->fem.fv->turb_omega;
  }

  if (pdgv[LIGHT_INTP]) {
    v = LIGHT_INTP;
    scalar_fv_fill(goma_ctx->fem.esp->poynt[0], goma_ctx->fem.esp_dot->poynt[0],
                   goma_ctx->fem.esp_old->poynt[0], goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->poynt[0]), &(goma_ctx->fem.fv_dot->poynt[0]),
                   &(goma_ctx->fem.fv_old->poynt[0]));
    stateVector[LIGHT_INTP] = goma_ctx->fem.fv->poynt[0];
  }
  if (pdgv[LIGHT_INTM]) {
    v = LIGHT_INTM;
    scalar_fv_fill(goma_ctx->fem.esp->poynt[1], goma_ctx->fem.esp_dot->poynt[1],
                   goma_ctx->fem.esp_old->poynt[1], goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->poynt[1]), &(goma_ctx->fem.fv_dot->poynt[1]),
                   &(goma_ctx->fem.fv_old->poynt[1]));
    stateVector[LIGHT_INTM] = goma_ctx->fem.fv->poynt[1];
  }
  if (pdgv[LIGHT_INTD]) {
    v = LIGHT_INTD;
    scalar_fv_fill(goma_ctx->fem.esp->poynt[2], goma_ctx->fem.esp_dot->poynt[2],
                   goma_ctx->fem.esp_old->poynt[2], goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->poynt[2]), &(goma_ctx->fem.fv_dot->poynt[2]),
                   &(goma_ctx->fem.fv_old->poynt[2]));
    stateVector[LIGHT_INTD] = goma_ctx->fem.fv->poynt[2];
  }
  if (pdgv[RESTIME]) {
    v = RESTIME;
    scalar_fv_fill(goma_ctx->fem.esp->restime, goma_ctx->fem.esp_dot->restime,
                   goma_ctx->fem.esp_old->restime, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->restime), &(goma_ctx->fem.fv_dot->restime),
                   &(goma_ctx->fem.fv_old->restime));
    stateVector[RESTIME] = goma_ctx->fem.fv->restime;
  }
  /*
   *	Porous sink mass
   */

  if (pdgv[POR_SINK_MASS]) {
    v = POR_SINK_MASS;
    scalar_fv_fill(goma_ctx->fem.esp->sink_mass, goma_ctx->fem.esp_dot->sink_mass,
                   goma_ctx->fem.esp_old->sink_mass, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->sink_mass), &(goma_ctx->fem.fv_dot->sink_mass),
                   &(goma_ctx->fem.fv_old->sink_mass));
    stateVector[POR_SINK_MASS] = goma_ctx->fem.fv->sink_mass;
  } /*else if (upd->vp[pg->imtrx][v] == -1) {
      fv->sink_mass = fv_old->sink_mass = fv_dot->sink_mass = 0.;
      }*/

  /*
   * Pressure...
   */

  if (pdgv[PRESSURE]) {
    v = PRESSURE;
    scalar_fv_fill(goma_ctx->fem.esp->P, goma_ctx->fem.esp_dot->P, goma_ctx->fem.esp_old->P,
                   goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->P), &(goma_ctx->fem.fv_dot->P), &(goma_ctx->fem.fv_old->P));
    stateVector[v] = goma_ctx->fem.fv->P + goma_ctx->fem.upd->Pressure_Datum;
  }

  if (pdgv[PSTAR]) {
    v = PSTAR;
    scalar_fv_fill(goma_ctx->fem.esp->P_star, goma_ctx->fem.esp_dot->P_star,
                   goma_ctx->fem.esp_old->P_star, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                   &(goma_ctx->fem.fv->P_star), &(goma_ctx->fem.fv_dot->P_star),
                   &(goma_ctx->fem.fv_old->P_star));
    stateVector[v] = goma_ctx->fem.fv->P_star + goma_ctx->fem.upd->Pressure_Datum;
  }

  if (pdgv[EM_CONT_REAL]) {
    v = EM_CONT_REAL;
    scalar_fv_fill(
        goma_ctx->fem.esp->epr, goma_ctx->fem.esp_dot->epr, goma_ctx->fem.esp_old->epr,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->epr), &(goma_ctx->fem.fv_dot->epr), &(goma_ctx->fem.fv_old->epr));
    stateVector[v] = goma_ctx->fem.fv->epr;
  }
  if (pdgv[EM_CONT_IMAG]) {
    v = EM_CONT_IMAG;
    scalar_fv_fill(
        goma_ctx->fem.esp->epi, goma_ctx->fem.esp_dot->epi, goma_ctx->fem.esp_old->epi,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->epi), &(goma_ctx->fem.fv_dot->epi), &(goma_ctx->fem.fv_old->epi));
    stateVector[v] = goma_ctx->fem.fv->epi;
  }
  /*
   *  Set the state vector pressure datum to include an additional
   *  uniform pressure datum.
   */

  /*
   * Mesh displacement (vector)...
   * and positions (vector)
   */

  /*
   * Insure default value of 3rd coordinate is zero for low
   * dimensional problems...DIM=3, while dim can be less...
   */

  for (p = 0; p < dim; p++) {
    goma_ctx->fem.fv->x0[p] = 0.0;
    goma_ctx->fem.fv->x[p] = 0.0;
    goma_ctx->fem.fv_old->x[p] = 0.0;
    goma_ctx->fem.fv_dot->x[p] = 0.0;
    goma_ctx->fem.fv_dot_old->x[p] = 0.0;

    goma_ctx->fem.fv->d[p] = 0.0;
    goma_ctx->fem.fv_old->d[p] = 0.0;
    goma_ctx->fem.fv_dot->d[p] = 0.0;
    goma_ctx->fem.fv_dot_old->d[p] = 0.0;

    if (goma_ctx->fem.tran->solid_inertia) {
      goma_ctx->fem.fv_dot_dot->x[p] = 0;
      goma_ctx->fem.fv_dot_dot->d[p] = 0;
    }

    /*
     * If this is a shell element, mesh displacements may not be
     * defined on this element block even if the mesh is deforming.
     * In this case, displacements are defined on a neighbor block
     * and ei[upd->matrix_index[v]]->deforming_mesh will be TRUE, so that the true
     * displaced coordinates can be loaded here.
     */
    if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->deforming_mesh) {
      /*
       * ShapeVar will always be mesh displacement where it is defined.
       * Otherwise (e.g. for shell elements), it will be set correctly.
       */
      bfv = goma_ctx->fem.bf[goma_ctx->fem.pd->ShapeVar];
      v = goma_ctx->fem.pd->ShapeVar;
      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

      for (i = 0; i < dofs; i++) {
        node = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof_list[R_MESH1][i];
        index = goma_ctx->mesh.proc_elem_connect
                    [goma_ctx->mesh.proc_connect_ptr
                         [goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->ielem] +
                     node];
        goma_ctx->fem.fv->d[p] += *goma_ctx->fem.esp->d[p][i] * bfv->phi[i];
        goma_ctx->fem.fv->x[p] +=
            (goma_ctx->mesh.coor[p][index] + *goma_ctx->fem.esp->d[p][i]) * bfv->phi[i];
        goma_ctx->fem.fv->x0[p] += goma_ctx->mesh.coor[p][index] * bfv->phi[i];

        /*
         * HKM -> We calculate the old xdot's by doing a little pointer
         *        arithmetic. Basically we use esp_dot as an offset to
         *        the base and then correct for the base address.
         */
        if (transient_run) {
          goma_ctx->fem.fv_old->d[p] += *goma_ctx->fem.esp_old->d[p][i] * bfv->phi[i];
          goma_ctx->fem.fv_old->x[p] +=
              (goma_ctx->mesh.coor[p][index] + *goma_ctx->fem.esp_old->d[p][i]) *
              goma_ctx->fem.bf[v]->phi[i];
          goma_ctx->fem.fv_dot->d[p] += *goma_ctx->fem.esp_dot->d[p][i] * bfv->phi[i];
          goma_ctx->fem.fv_dot->x[p] += *goma_ctx->fem.esp_dot->d[p][i] * bfv->phi[i];
          if (goma_ctx->fem.tran->solid_inertia) {
            goma_ctx->fem.fv_dot_dot->d[p] += *goma_ctx->fem.esp_dbl_dot->d[p][i] * bfv->phi[i];
            goma_ctx->fem.fv_dot_dot->x[p] += *goma_ctx->fem.esp_dbl_dot->d[p][i] * bfv->phi[i];
          }

          if (goma_ctx->fem.upd->Total_Num_Matrices > 1) {
            goma_ctx->fem.fv_dot_old->x[p] +=
                *(goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot_old -
                  goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot +
                  goma_ctx->fem.esp_dot->d[p][i]) *
                bfv->phi[i];
            goma_ctx->fem.fv_dot_old->d[p] +=
                *(goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot_old -
                  goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot +
                  goma_ctx->fem.esp_dot->d[p][i]) *
                bfv->phi[i];
          } else {
            goma_ctx->fem.fv_dot_old->x[p] +=
                *(xdot_old_static - xdot_static + goma_ctx->fem.esp_dot->d[p][i]) * bfv->phi[i];
            goma_ctx->fem.fv_dot_old->d[p] +=
                *(xdot_old_static - xdot_static + goma_ctx->fem.esp_dot->d[p][i]) * bfv->phi[i];
          }
          if (goma_ctx->fem.tran->solid_inertia) {
            goma_ctx->fem.fv_dot_dot_old->d[p] +=
                *(x_dbl_dot_old_static - x_dbl_dot_static + goma_ctx->fem.esp_dbl_dot->d[p][i]) *
                bfv->phi[i];
          }
        } else {
          goma_ctx->fem.fv_old->d[p] = goma_ctx->fem.fv->d[p];
          goma_ctx->fem.fv_old->x[p] =
              goma_ctx->fem.fv->x[p]; /* Fixed grid stays fixed thru time. */
        }
      }
    }

    else /* Zero these only if not using mesh displacements: */
    {
      v = goma_ctx->fem.pd->ShapeVar;
      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

      if (goma_ctx->fem.bf[v]->interpolation == I_N1) {
        dofs = goma_ctx->fem.bf[v]->shape_dof;
      }

      goma_ctx->fem.fv->d[p] = 0.0;
      goma_ctx->fem.fv_old->d[p] = 0.0;
      goma_ctx->fem.fv_dot->d[p] = 0.0;
      goma_ctx->fem.fv_dot_dot->d[p] = 0.0;

      goma_ctx->fem.fv->x[p] = 0.;
      for (i = 0; i < dofs; i++) {
        if (goma_ctx->fem.bf[v]->interpolation == I_N1) {
          node = i;
        } else {
          node = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof_list[v][i];
        }
        index = goma_ctx->mesh.proc_elem_connect
                    [goma_ctx->mesh.proc_connect_ptr
                         [goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->ielem] +
                     node];
        goma_ctx->fem.fv->x[p] += (goma_ctx->mesh.coor[p][index]) * goma_ctx->fem.bf[v]->phi[i];
      }
      goma_ctx->fem.fv_old->x[p] = goma_ctx->fem.fv->x[p]; /* Fixed grid stays fixed thru time. */
      goma_ctx->fem.fv->x0[p] = goma_ctx->fem.fv->x[p];
      goma_ctx->fem.fv_dot->x[p] = 0.0;
      goma_ctx->fem.fv_dot_dot->x[p] = 0.0;
    }
  }

  /*
   * SOLID displacement (vector)...
   * and positions (vector)
   */
  for (p = 0; pdgv[SOLID_DISPLACEMENT1] && p < dim; p++) {
    v = SOLID_DISPLACEMENT1 + p;
    goma_ctx->fem.fv->d_rs[p] = 0.;
    goma_ctx->fem.fv_old->d_rs[p] = 0.;
    goma_ctx->fem.fv_dot->d_rs[p] = 0.;
    goma_ctx->fem.fv_dot_dot->d_rs[p] = 0.;
    goma_ctx->fem.fv_dot_old->d_rs[p] = 0.;
    goma_ctx->fem.fv_dot_dot_old->d_rs[p] = 0.;

    if (pdgv[v]) {
      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

      for (i = 0; i < dofs; i++) {
        node = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof_list[R_SOLID1][i];
        index = goma_ctx->mesh.proc_elem_connect
                    [goma_ctx->mesh.proc_connect_ptr
                         [goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->ielem] +
                     node];
        goma_ctx->fem.fv->d_rs[p] += *goma_ctx->fem.esp->d_rs[p][i] * goma_ctx->fem.bf[v]->phi[i];

        if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
          goma_ctx->fem.fv_old->d_rs[p] +=
              *goma_ctx->fem.esp_old->d_rs[p][i] * goma_ctx->fem.bf[v]->phi[i];
          goma_ctx->fem.fv_dot->d_rs[p] +=
              *goma_ctx->fem.esp_dot->d_rs[p][i] * goma_ctx->fem.bf[v]->phi[i];
          if (goma_ctx->fem.tran->solid_inertia) {
            goma_ctx->fem.fv_dot_dot->d_rs[p] +=
                *goma_ctx->fem.esp_dbl_dot->d_rs[p][i] * goma_ctx->fem.bf[v]->phi[i];
          }

          if (goma_ctx->fem.upd->Total_Num_Matrices > 1) {
            goma_ctx->fem.fv_dot_old->d_rs[p] +=
                *(goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot_old -
                  goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot +
                  goma_ctx->fem.esp_dot->d_rs[p][i]) *
                goma_ctx->fem.bf[v]->phi[i];
          } else {
            goma_ctx->fem.fv_dot_old->d_rs[p] +=
                *(xdot_old_static - xdot_static + goma_ctx->fem.esp_dot->d_rs[p][i]) *
                goma_ctx->fem.bf[v]->phi[i];
          }

          if (goma_ctx->fem.tran->solid_inertia) {
            goma_ctx->fem.fv_dot_dot_old->d_rs[p] +=
                *(x_dbl_dot_old_static - x_dbl_dot_static + goma_ctx->fem.esp_dbl_dot->d_rs[p][i]) *
                goma_ctx->fem.bf[v]->phi[i];
          }
        } else {
          goma_ctx->fem.fv_old->d_rs[p] = goma_ctx->fem.fv->d_rs[p];
        }
      }
    }
  }

  status = load_coordinate_scales(goma_ctx->fem.pd->CoordinateSystem, goma_ctx->fem.fv);
  GOMA_EH(status, "load_coordinate_scales(fv)");

  /*
   * Velocity (vector)...
   */

  /*
   * Default: all velocities are zero...
   */
  for (p = 0; p < WIM; p++) {
    v = VELOCITY1 + p;
    if (pdgv[v]) {
      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
      goma_ctx->fem.fv->v[p] = 0.;
      goma_ctx->fem.fv_old->v[p] = 0.;
      goma_ctx->fem.fv_dot->v[p] = 0.;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->v[p] += *goma_ctx->fem.esp->v[p][i] * goma_ctx->fem.bf[v]->phi[i];
        if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
          goma_ctx->fem.fv_old->v[p] +=
              *goma_ctx->fem.esp_old->v[p][i] * goma_ctx->fem.bf[v]->phi[i];
          goma_ctx->fem.fv_dot->v[p] +=
              *goma_ctx->fem.esp_dot->v[p][i] * goma_ctx->fem.bf[v]->phi[i];
        }
      }
    }
    stateVector[VELOCITY1 + p] = goma_ctx->fem.fv->v[p];
  }

  for (p = 0; p < WIM; p++) {
    v = USTAR + p;
    if (pdgv[v]) {
      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
      goma_ctx->fem.fv->v_star[p] = 0.;
      goma_ctx->fem.fv_old->v_star[p] = 0.;
      goma_ctx->fem.fv_dot->v_star[p] = 0.;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->v_star[p] +=
            *goma_ctx->fem.esp->v_star[p][i] * goma_ctx->fem.bf[v]->phi[i];
        if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
          goma_ctx->fem.fv_old->v_star[p] +=
              *goma_ctx->fem.esp_old->v_star[p][i] * goma_ctx->fem.bf[v]->phi[i];
          goma_ctx->fem.fv_dot->v_star[p] +=
              *goma_ctx->fem.esp_dot->v_star[p][i] * goma_ctx->fem.bf[v]->phi[i];
        }
      }
    }
    stateVector[VELOCITY1 + p] = goma_ctx->fem.fv->v_star[p];
  }

  /*
   * Particle velocity (vector)...
   */

  /*
   * Default: all velocities are zero...
   */
  /*  for ( p=WIM; p<DIM; p++)
      {
      v = PVELOCITY1 + p;
      if ( pd->v[pg->imtrx][v] || (upd->vp[pg->imtrx][v] == -1) )
      {
      fv->pv[p]     = 0.;
      fv_old->pv[p] = 0.;
      fv_dot->pv[p] = 0.;
      }
      } */

  for (p = 0; pdgv[PVELOCITY1] && p < WIM; p++) {
    v = PVELOCITY1 + p;
    if (pdgv[v]) {
      goma_ctx->fem.fv->pv[p] = 0.0;
      goma_ctx->fem.fv_old->pv[p] = 0.0;
      goma_ctx->fem.fv_dot->pv[p] = 0.0;

      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->pv[p] += *goma_ctx->fem.esp->pv[p][i] * goma_ctx->fem.bf[v]->phi[i];

        if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
          goma_ctx->fem.fv_old->pv[p] +=
              *goma_ctx->fem.esp_old->pv[p][i] * goma_ctx->fem.bf[v]->phi[i];
          goma_ctx->fem.fv_dot->pv[p] +=
              *goma_ctx->fem.esp_dot->pv[p][i] * goma_ctx->fem.bf[v]->phi[i];
        }
      }
    }
  }

  /* Extension velocity */

  v = EXT_VELOCITY;
  if (pdgv[EXT_VELOCITY]) {
    scalar_fv_fill(
        goma_ctx->fem.esp->ext_v, goma_ctx->fem.esp_dot->ext_v, goma_ctx->fem.esp_old->ext_v,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->ext_v), &(goma_ctx->fem.fv_dot->ext_v), &(goma_ctx->fem.fv_old->ext_v));
    stateVector[EXT_VELOCITY] = goma_ctx->fem.fv->ext_v;
  }

  /* Electric Field */

  /*
   * Default: all velocities are zero...
   */
  for (p = 0; pdgv[EFIELD1] && p < WIM; p++) {
    v = EFIELD1 + p;

    if (pdgv[v]) {
      goma_ctx->fem.fv->E_field[p] = 0.;

      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->E_field[p] +=
            *goma_ctx->fem.esp->E_field[p][i] * goma_ctx->fem.bf[v]->phi[i];
      }
    }
  }

  /* Phase functions */

  for (p = 0; pdgv[PHASE1] && p < goma_ctx->fem.pfd->num_phase_funcs; p++) {
    v = PHASE1 + p;
    if (pdgv[v]) {
      scalar_fv_fill(goma_ctx->fem.esp->pF[p], goma_ctx->fem.esp_dot->pF[p],
                     goma_ctx->fem.esp_old->pF[p], goma_ctx->fem.bf[v]->phi,
                     goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                     &(goma_ctx->fem.fv->pF[p]), &(goma_ctx->fem.fv_dot->pF[p]),
                     &(goma_ctx->fem.fv_old->pF[p]));
    }
  }

  /*
   * Polymer Stress (tensor)...
   */

  for (mode = 0; pdgv[POLYMER_STRESS11] && mode < goma_ctx->fem.vn->modes; mode++) {
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        v = v_s[mode][p][q];
        if (pdgv[v] || (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][v] == -1)) {
          /* good default behavior */
          goma_ctx->fem.fv->S[mode][p][q] = 0.;
          goma_ctx->fem.fv_old->S[mode][p][q] = 0.;
          goma_ctx->fem.fv_dot->S[mode][p][q] = 0.;
        }
      }
    }
  }

  for (mode = 0; pdgv[POLYMER_STRESS11] && mode < goma_ctx->fem.vn->modes; mode++) {
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        if (p <= q) {
          v = v_s[mode][p][q];
          if (pdgv[v]) {
            dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
            for (i = 0; i < dofs; i++) {
              goma_ctx->fem.fv->S[mode][p][q] +=
                  *goma_ctx->fem.esp->S[mode][p][q][i] * goma_ctx->fem.bf[v]->phi[i];
              if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
                goma_ctx->fem.fv_old->S[mode][p][q] +=
                    *goma_ctx->fem.esp_old->S[mode][p][q][i] * goma_ctx->fem.bf[v]->phi[i];
                goma_ctx->fem.fv_dot->S[mode][p][q] +=
                    *goma_ctx->fem.esp_dot->S[mode][p][q][i] * goma_ctx->fem.bf[v]->phi[i];
              }
            }
          }
          /* form the entire symmetric stress matrix for the momentum equation */
          goma_ctx->fem.fv->S[mode][q][p] = goma_ctx->fem.fv->S[mode][p][q];
          goma_ctx->fem.fv_old->S[mode][q][p] = goma_ctx->fem.fv_old->S[mode][p][q];
          goma_ctx->fem.fv_dot->S[mode][q][p] = goma_ctx->fem.fv_dot->S[mode][p][q];
        }
      }
    }
  }

  /*
   * Velocity Gradient (tensor)...
   */

  for (p = 0; pdgv[VELOCITY_GRADIENT11] && p < VIM; p++) {
    if (goma_ctx->fem.gn->ConstitutiveEquation == BINGHAM_MIXED) {
      for (q = 0; q < VIM; q++) {
        if (q >= p) {
          v = v_g[p][q];
          if (pdgv[v]) {
            goma_ctx->fem.fv->G[p][q] = goma_ctx->fem.fv_old->G[p][q] =
                goma_ctx->fem.fv_dot->G[p][q] = 0.0;
            if (q > p) {
              goma_ctx->fem.fv->G[q][p] = goma_ctx->fem.fv_old->G[q][p] =
                  goma_ctx->fem.fv_dot->G[q][p] = 0.0;
            }
            dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
            for (i = 0; i < dofs; i++) {
              goma_ctx->fem.fv->G[p][q] +=
                  *goma_ctx->fem.esp->G[p][q][i] * goma_ctx->fem.bf[v]->phi[i];
              if (q > p) {
                goma_ctx->fem.fv->G[q][p] +=
                    *goma_ctx->fem.esp->G[p][q][i] * goma_ctx->fem.bf[v]->phi[i];
              }
              if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
                goma_ctx->fem.fv_old->G[p][q] +=
                    *goma_ctx->fem.esp_old->G[p][q][i] * goma_ctx->fem.bf[v]->phi[i];
                if (q > p) {
                  goma_ctx->fem.fv_dot->G[q][p] +=
                      *goma_ctx->fem.esp_dot->G[p][q][i] * goma_ctx->fem.bf[v]->phi[i];
                }
              }
            }
          }
        }
      }
    } else {
      for (q = 0; q < VIM; q++) {
        v = v_g[p][q];
        if (pdgv[v]) {
          goma_ctx->fem.fv->G[p][q] = goma_ctx->fem.fv_old->G[p][q] =
              goma_ctx->fem.fv_dot->G[p][q] = 0.0;
          dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->G[p][q] +=
                *goma_ctx->fem.esp->G[p][q][i] * goma_ctx->fem.bf[v]->phi[i];
            if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
              goma_ctx->fem.fv_old->G[p][q] +=
                  *goma_ctx->fem.esp_old->G[p][q][i] * goma_ctx->fem.bf[v]->phi[i];
              goma_ctx->fem.fv_dot->G[p][q] +=
                  *goma_ctx->fem.esp_dot->G[p][q][i] * goma_ctx->fem.bf[v]->phi[i];
            }
          }
        }
      }
    }
  }

  /*
   * Species Unknown Variable
   *      Modifications for non-dilute systems:
   *      -> We need to calculate the mole fraction of the last species
   *         in the mechanism, even if there isn't an equation
   *         for it. This is done via the sum MF = 1
   *         constraint, here. Generalization to other equations
   *         of state should be added here in the future.
   */

  if (pdgv[MASS_FRACTION]) {
    v = MASS_FRACTION;
    if (goma_ctx->fem.pd->Num_Species_Eqn != goma_ctx->fem.pd->Num_Species) {
      N = goma_ctx->fem.pd->Num_Species - 1;

      goma_ctx->fem.fv->c[N] = goma_ctx->fem.fv_old->c[N] = goma_ctx->fem.fv_dot->c[N] = 0.;
      for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
        scalar_fv_fill(goma_ctx->fem.esp->c[w], goma_ctx->fem.esp_dot->c[w],
                       goma_ctx->fem.esp_old->c[w], goma_ctx->fem.bf[v]->phi,
                       goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                       &(goma_ctx->fem.fv->c[w]), &(goma_ctx->fem.fv_dot->c[w]),
                       &(goma_ctx->fem.fv_old->c[w]));
        goma_ctx->fem.fv->c[N] -= goma_ctx->fem.fv->c[w];
        goma_ctx->fem.fv_old->c[N] -= goma_ctx->fem.fv_old->c[w];
        goma_ctx->fem.fv_dot->c[N] -= goma_ctx->fem.fv_dot->c[w];
        stateVector[SPECIES_UNK_0 + w] = goma_ctx->fem.fv->c[w];
      }
      switch (goma_ctx->fem.mp->Species_Var_Type) {
      case SPECIES_UNDEFINED_FORM:
      case SPECIES_MOLE_FRACTION:
      case SPECIES_MASS_FRACTION:
      case SPECIES_VOL_FRACTION:
        goma_ctx->fem.fv->c[N] += 1.0;
        stateVector[SPECIES_UNK_0 + N] = goma_ctx->fem.fv->c[N];
        break;
      case SPECIES_CONCENTRATION:
        /*
         *  HKM:  These are currently problematic cases that
         *  argue for including the last species as a dependent
         *  variable in the solution vector. The reason is that
         *  a small nonlinear iteration must be performed here
         *  for calculation of the total species concentration
         *  and the species concentration of the last species
         *  in the mechanism. An equation of state has to
         *  be assumed, and then inverted.
         *
         *   C = Sum_i=1,N[C_i],
         *
         *  where C is determined from the equation of state
         *  as a function of T, P, [other_state_variables], and
         *  C_1 to C_N.
         *
         *  If the problem is
         *  dilute, then the whole process is circumvented:
         *     C = C_N.
         *  or at least C isn't a function of the last species
         *  in the mechanism:
         *     C = C(T, P, C_1, ..., C_N-1)
         *
         *  The dilute case is carried out below. However,
         *  note that c[N] = 0 is assumed in the function
         *  call. If c[N] is used, then this routine will
         *  give the wrong answer, until a nonlinear iteration
         *  loop is installed below.
         *
         * -> The same is true for the SPECIES_DENSITY
         *    loop.
         */
        switch (goma_ctx->fem.mp->DensityModel) {
        case DENSITY_CONSTANT_LAST_CONC:
          goma_ctx->fem.fv->c[N] = goma_ctx->fem.mp->u_density[0];
          break;
        default:
          rho = calc_concentration(goma_ctx->fem.mp, FALSE, NULL);
          goma_ctx->fem.fv->c[N] += rho;
        }
        stateVector[SPECIES_UNK_0 + N] = goma_ctx->fem.fv->c[N];
        break;
      case SPECIES_DENSITY:
        /* note, this won't work for time dependent density models, but I got tired ... RRR*/
        rho = calc_density(goma_ctx->fem.mp, FALSE, NULL, 0.);
        goma_ctx->fem.fv->c[N] += rho;
        stateVector[SPECIES_UNK_0 + N] = goma_ctx->fem.fv->c[N];
        break;
      case SPECIES_CAP_PRESSURE:
        break;
      default:
        break;
      }
      goma_ctx->fem.mp->StateVector_speciesVT = goma_ctx->fem.upd->Species_Var_Type;
    } else {
      for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
        scalar_fv_fill(goma_ctx->fem.esp->c[w], goma_ctx->fem.esp_dot->c[w],
                       goma_ctx->fem.esp_old->c[w], goma_ctx->fem.bf[v]->phi,
                       goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                       &(goma_ctx->fem.fv->c[w]), &(goma_ctx->fem.fv_dot->c[w]),
                       &(goma_ctx->fem.fv_old->c[w]));
        goma_ctx->fem.fv_dot_old->c[w] = 0;
        for (int i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v]; i++) {
          goma_ctx->fem.fv_dot_old->c[w] +=
              *(goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot_old -
                goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot +
                goma_ctx->fem.esp_dot->c[w][i]) *
              goma_ctx->fem.bf[v]->phi[i];
        }
        stateVector[SPECIES_UNK_0 + w] = goma_ctx->fem.fv->c[w];
      }
    }
  } /* HKM->
     *  (we may want to insert reference concentrations here
     *   if available)
     */

  /*else if (upd->vp[pg->imtrx][v] == -1) {

  for (w = 0; w < pd->Num_Species; w++) {
  fv->c[w]     = 0.;
  fv_old->c[w] = 0.;
  fv_dot->c[w] = 0.;
  stateVector[SPECIES_UNK_0+w] = fv->c[w];
  }
  } */

  /*
   * Porous media Variables
   */

  /*  if (pd->v[pg->imtrx][v] || (upd->vp[pg->imtrx][v] == -1) )
      {
      fv->p_liq=0.;
      fv_old->p_liq = 0.;
      fv_dot->p_liq = 0.;
      fv_dot_old->p_liq = 0.0;
      } */

  if (pdgv[POR_LIQ_PRES]) {
    v = POR_LIQ_PRES;
    goma_ctx->fem.fv->p_liq = 0.;
    goma_ctx->fem.fv_old->p_liq = 0.;
    goma_ctx->fem.fv_dot->p_liq = 0.;
    goma_ctx->fem.fv_dot_old->p_liq = 0.0;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (i = 0; i < dofs; i++) {
      goma_ctx->fem.fv->p_liq += *goma_ctx->fem.esp->p_liq[i] * goma_ctx->fem.bf[v]->phi[i];
      if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
        goma_ctx->fem.fv_old->p_liq +=
            *goma_ctx->fem.esp_old->p_liq[i] * goma_ctx->fem.bf[v]->phi[i];
        goma_ctx->fem.fv_dot->p_liq +=
            *goma_ctx->fem.esp_dot->p_liq[i] * goma_ctx->fem.bf[v]->phi[i];

        if (goma_ctx->fem.upd->Total_Num_Matrices > 1) {
          goma_ctx->fem.fv_dot_old->p_liq +=
              *(goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot_old -
                goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot +
                goma_ctx->fem.esp_dot->p_liq[i]) *
              goma_ctx->fem.bf[v]->phi[i];
        } else {
          goma_ctx->fem.fv_dot_old->p_liq +=
              *(xdot_old_static - xdot_static + goma_ctx->fem.esp_dot->p_liq[i]) *
              goma_ctx->fem.bf[v]->phi[i];
        }
      }
    }
  }

  /*  if (pd->v[pg->imtrx][v] || (upd->vp[pg->imtrx][v] == -1) )
      {
      fv->p_gas=0.;
      fv_old->p_gas = 0.;
      fv_dot->p_gas = 0.;
      fv_dot_old->p_gas = 0.0;
      }
  */
  if (pdgv[POR_GAS_PRES]) {
    v = POR_GAS_PRES;
    goma_ctx->fem.fv->p_gas = 0.;
    goma_ctx->fem.fv_old->p_gas = 0.;
    goma_ctx->fem.fv_dot->p_gas = 0.;
    goma_ctx->fem.fv_dot_old->p_gas = 0.0;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (i = 0; i < dofs; i++) {
      goma_ctx->fem.fv->p_gas += *goma_ctx->fem.esp->p_gas[i] * goma_ctx->fem.bf[v]->phi[i];
      if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
        goma_ctx->fem.fv_old->p_gas +=
            *goma_ctx->fem.esp_old->p_gas[i] * goma_ctx->fem.bf[v]->phi[i];
        goma_ctx->fem.fv_dot->p_gas +=
            *goma_ctx->fem.esp_dot->p_gas[i] * goma_ctx->fem.bf[v]->phi[i];
        if (goma_ctx->fem.upd->Total_Num_Matrices > 1) {
          goma_ctx->fem.fv_dot_old->p_gas +=
              *(goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot_old -
                goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot +
                goma_ctx->fem.esp_dot->p_gas[i]) *
              goma_ctx->fem.bf[v]->phi[i];
        } else {
          goma_ctx->fem.fv_dot_old->p_gas +=
              *(xdot_old_static - xdot_static + goma_ctx->fem.esp_dot->p_gas[i]) *
              goma_ctx->fem.bf[v]->phi[i];
        }
      }
    }
  }

  /*  if (pd->v[pg->imtrx][v] || (upd->vp[pg->imtrx][v] == -1) )
      {
      fv->porosity=0.;
      fv_old->porosity = 0.;
      fv_dot->porosity = 0.;
      fv_dot_old->porosity = 0.0;
      }
  */
  if (pdgv[POR_POROSITY]) {
    v = POR_POROSITY;
    goma_ctx->fem.fv->porosity = 0.;
    goma_ctx->fem.fv_old->porosity = 0.;
    goma_ctx->fem.fv_dot->porosity = 0.;
    goma_ctx->fem.fv_dot_old->porosity = 0.0;

    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (i = 0; i < dofs; i++) {
      goma_ctx->fem.fv->porosity += *goma_ctx->fem.esp->porosity[i] * goma_ctx->fem.bf[v]->phi[i];
      if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
        goma_ctx->fem.fv_old->porosity +=
            *goma_ctx->fem.esp_old->porosity[i] * goma_ctx->fem.bf[v]->phi[i];
        goma_ctx->fem.fv_dot->porosity +=
            *goma_ctx->fem.esp_dot->porosity[i] * goma_ctx->fem.bf[v]->phi[i];
        if (goma_ctx->fem.upd->Total_Num_Matrices > 1) {
          goma_ctx->fem.fv_dot_old->porosity +=
              *(goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot_old -
                goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot +
                goma_ctx->fem.esp_dot->porosity[i]) *
              goma_ctx->fem.bf[v]->phi[i];
        } else {
          goma_ctx->fem.fv_dot_old->porosity +=
              *(xdot_old_static - xdot_static + goma_ctx->fem.esp_dot->porosity[i]) *
              goma_ctx->fem.bf[v]->phi[i];
        }
      }
    }
  }

  if (pdgv[POR_TEMP]) {
    v = POR_TEMP;
    goma_ctx->fem.fv->T = 0.;
    goma_ctx->fem.fv_old->T = 0.;
    goma_ctx->fem.fv_dot->T = 0.;
    goma_ctx->fem.fv_dot_old->T = 0.0;

    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (i = 0; i < dofs; i++) {
      goma_ctx->fem.fv->T += *goma_ctx->fem.esp->T[i] * goma_ctx->fem.bf[v]->phi[i];
      if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
        goma_ctx->fem.fv_old->T += *goma_ctx->fem.esp_old->T[i] * goma_ctx->fem.bf[v]->phi[i];
        goma_ctx->fem.fv_dot->T += *goma_ctx->fem.esp_dot->T[i] * goma_ctx->fem.bf[v]->phi[i];

        if (goma_ctx->fem.upd->Total_Num_Matrices > 1) {
          goma_ctx->fem.fv_dot_old->T +=
              *(goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot_old -
                goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot +
                goma_ctx->fem.esp_dot->T[i]) *
              goma_ctx->fem.bf[v]->phi[i];
        } else {
          goma_ctx->fem.fv_dot_old->T +=
              *(xdot_old_static - xdot_static + goma_ctx->fem.esp_dot->T[i]) *
              goma_ctx->fem.bf[v]->phi[i];
        }
      }
    }
  }

  /*
   * Vorticity principle flow direction
   */
  for (p = 0; pdgv[VORT_DIR1] && p < DIM; p++) {
    v = VORT_DIR1 + p;
    /*if (pd->v[pg->imtrx][v] || upd->vp[pg->imtrx][v] == -1) fv->vd[p] = 0.0; */

    if (pdgv[v]) {
      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
      goma_ctx->fem.fv->vd[p] = 0.0;
      for (i = 0; i < dofs; i++)
        goma_ctx->fem.fv->vd[p] += *goma_ctx->fem.esp->vd[p][i] * goma_ctx->fem.bf[v]->phi[i];
    }
  }

  /*
   * Lagrange Multiplier Field
   */
  for (p = 0; pdgv[LAGR_MULT1] && p < DIM; p++) {
    v = LAGR_MULT1 + p;
    /*      if (pd->v[pg->imtrx][v] || upd->vp[pg->imtrx][v] == -1)
            {
            fv->lm[p] = 0.0;
            fv_old->lm[p] = 0.0;
            }
    */
    if (pdgv[v]) {
      goma_ctx->fem.fv->lm[p] = 0.0;
      goma_ctx->fem.fv_old->lm[p] = 0.0;

      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->lm[p] += *goma_ctx->fem.esp->lm[p][i] * goma_ctx->fem.bf[v]->phi[i];
        if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
          goma_ctx->fem.fv_old->lm[p] +=
              *goma_ctx->fem.esp_old->lm[p][i] * goma_ctx->fem.bf[v]->phi[i];
        }
      }
    }
  }

  /*
   * Eigenvalue associated with vd.
   */

  /*  if (pd->v[pg->imtrx][v] || upd->vp[pg->imtrx][v] == -1) fv->vlambda = 0.0; */

  if (pdgv[VORT_LAMBDA]) {
    v = VORT_LAMBDA;
    goma_ctx->fem.fv->vlambda = 0.0;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (i = 0; i < dofs; i++)
      goma_ctx->fem.fv->vlambda += *goma_ctx->fem.esp->vlambda[i] * goma_ctx->fem.bf[v]->phi[i];
  }

  for (p = 0; pdgv[MOMENT0] && p < MAX_MOMENTS; p++) {
    v = MOMENT0 + p;
    if (pdgv[v]) {
      scalar_fv_fill(goma_ctx->fem.esp->moment[p], goma_ctx->fem.esp_dot->moment[p],
                     goma_ctx->fem.esp_old->moment[p], goma_ctx->fem.bf[v]->phi,
                     goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
                     &(goma_ctx->fem.fv->moment[p]), &(goma_ctx->fem.fv_dot->moment[p]),
                     &(goma_ctx->fem.fv_old->moment[p]));

      goma_ctx->fem.fv_dot_old->moment[p] = 0;
      for (int i = 0; i < goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v]; i++) {
        if (goma_ctx->fem.upd->Total_Num_Matrices > 1) {
          goma_ctx->fem.fv_dot_old->moment[p] +=
              *(goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot_old -
                goma_ctx->fem.pg->matrices[goma_ctx->fem.upd->matrix_index[v]].xdot +
                goma_ctx->fem.esp_dot->moment[p][i]) *
              goma_ctx->fem.bf[v]->phi[i];
        } else {
          goma_ctx->fem.fv_dot_old->moment[p] +=
              *(xdot_old_static - xdot_static + goma_ctx->fem.esp_dot->moment[p][i]) *
              goma_ctx->fem.bf[v]->phi[i];
        }
      }
    }
  }

  if (pdgv[DENSITY_EQN]) {
    v = DENSITY_EQN;
    scalar_fv_fill(
        goma_ctx->fem.esp->rho, goma_ctx->fem.esp_dot->rho, goma_ctx->fem.esp_old->rho,
        goma_ctx->fem.bf[v]->phi, goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v],
        &(goma_ctx->fem.fv->rho), &(goma_ctx->fem.fv_dot->rho), &(goma_ctx->fem.fv_old->rho));
    stateVector[v] = goma_ctx->fem.fv->rho;
  }

  if (pdgv[TFMP_PRES]) {

    v = TFMP_PRES;
    scalar_fv_fill(goma_ctx->fem.esp->tfmp_pres, goma_ctx->fem.esp_dot->tfmp_pres,
                   goma_ctx->fem.esp_old->tfmp_pres, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v],
                   &(goma_ctx->fem.fv->tfmp_pres), &(goma_ctx->fem.fv_dot->tfmp_pres),
                   &(goma_ctx->fem.fv_old->tfmp_pres));
    stateVector[v] = goma_ctx->fem.fv->tfmp_pres;
  }
  if (pdgv[TFMP_SAT]) {
    v = TFMP_SAT;
    scalar_fv_fill(goma_ctx->fem.esp->tfmp_sat, goma_ctx->fem.esp_dot->tfmp_sat,
                   goma_ctx->fem.esp_old->tfmp_sat, goma_ctx->fem.bf[v]->phi,
                   goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v], &(goma_ctx->fem.fv->tfmp_sat),
                   &(goma_ctx->fem.fv_dot->tfmp_sat), &(goma_ctx->fem.fv_old->tfmp_sat));
    stateVector[v] = goma_ctx->fem.fv->tfmp_sat;
  }

  /*
   * EM Wave Vectors...
   */
  if (pdgv[EM_E1_REAL] || pdgv[EM_E2_REAL] || pdgv[EM_E3_REAL]) {
    v = EM_E1_REAL;
    if (goma_ctx->fem.bf[v]->interpolation == I_N1) {
      for (p = 0; p < DIM; p++) {
        goma_ctx->fem.fv->em_er[p] = 0.0;
        goma_ctx->fem.fv_old->em_er[p] = 0.0;
        goma_ctx->fem.fv_dot->em_er[p] = 0.0;

        if (pdgv[v]) {
          dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->em_er[p] +=
                *goma_ctx->fem.esp->em_er[0][i] * goma_ctx->fem.bf[v]->phi_e[i][p];

            if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
              goma_ctx->fem.fv_old->em_er[p] +=
                  *goma_ctx->fem.esp_old->em_er[0][i] * goma_ctx->fem.bf[v]->phi_e[i][p];
              goma_ctx->fem.fv_dot->em_er[p] +=
                  *goma_ctx->fem.esp_dot->em_er[0][i] * goma_ctx->fem.bf[v]->phi_e[i][p];
            }
          }
        }
      }

    } else {
      for (p = 0; p < DIM; p++) {
        v = EM_E1_REAL + p;
        goma_ctx->fem.fv->em_er[p] = 0.0;
        goma_ctx->fem.fv_old->em_er[p] = 0.0;
        goma_ctx->fem.fv_dot->em_er[p] = 0.0;

        if (pdgv[v]) {
          dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->em_er[p] +=
                *goma_ctx->fem.esp->em_er[p][i] * goma_ctx->fem.bf[v]->phi[i];

            if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
              goma_ctx->fem.fv_old->em_er[p] +=
                  *goma_ctx->fem.esp_old->em_er[p][i] * goma_ctx->fem.bf[v]->phi[i];
              goma_ctx->fem.fv_dot->em_er[p] +=
                  *goma_ctx->fem.esp_dot->em_er[p][i] * goma_ctx->fem.bf[v]->phi[i];
            }
          }
        }
      }
    }
  }

  if (pdgv[EM_E1_IMAG] || pdgv[EM_E2_IMAG] || pdgv[EM_E3_IMAG]) {
    v = EM_E1_IMAG;
    if (goma_ctx->fem.bf[v]->interpolation == I_N1) {
      for (p = 0; p < DIM; p++) {
        goma_ctx->fem.fv->em_ei[p] = 0.0;
        goma_ctx->fem.fv_old->em_ei[p] = 0.0;
        goma_ctx->fem.fv_dot->em_ei[p] = 0.0;

        if (pdgv[v]) {
          dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->em_ei[p] +=
                *goma_ctx->fem.esp->em_ei[0][i] * goma_ctx->fem.bf[v]->phi_e[i][p];

            if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
              goma_ctx->fem.fv_old->em_ei[p] +=
                  *goma_ctx->fem.esp_old->em_ei[0][i] * goma_ctx->fem.bf[v]->phi_e[i][p];
              goma_ctx->fem.fv_dot->em_ei[p] +=
                  *goma_ctx->fem.esp_dot->em_ei[0][i] * goma_ctx->fem.bf[v]->phi_e[i][p];
            }
          }
        }
      }

    } else {
      for (p = 0; p < DIM; p++) {
        v = EM_E1_IMAG + p;
        goma_ctx->fem.fv->em_ei[p] = 0.0;
        goma_ctx->fem.fv_old->em_ei[p] = 0.0;
        goma_ctx->fem.fv_dot->em_ei[p] = 0.0;

        if (pdgv[v]) {
          dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->em_ei[p] +=
                *goma_ctx->fem.esp->em_ei[p][i] * goma_ctx->fem.bf[v]->phi[i];

            if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
              goma_ctx->fem.fv_old->em_ei[p] +=
                  *goma_ctx->fem.esp_old->em_ei[p][i] * goma_ctx->fem.bf[v]->phi[i];
              goma_ctx->fem.fv_dot->em_ei[p] +=
                  *goma_ctx->fem.esp_dot->em_ei[p][i] * goma_ctx->fem.bf[v]->phi[i];
            }
          }
        }
      }
    }
  }

  if (pdgv[EM_H1_REAL] || pdgv[EM_H2_REAL] || pdgv[EM_H3_REAL]) {
    for (p = 0; p < DIM; p++) {
      v = EM_H1_REAL + p;
      goma_ctx->fem.fv->em_hr[p] = 0.0;
      goma_ctx->fem.fv_old->em_hr[p] = 0.0;
      goma_ctx->fem.fv_dot->em_hr[p] = 0.0;
      if (pdgv[v]) {
        dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
        for (i = 0; i < dofs; i++) {
          goma_ctx->fem.fv->em_hr[p] +=
              *goma_ctx->fem.esp->em_hr[p][i] * goma_ctx->fem.bf[v]->phi[i];

          if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
            goma_ctx->fem.fv_old->em_hr[p] +=
                *goma_ctx->fem.esp_old->em_hr[p][i] * goma_ctx->fem.bf[v]->phi[i];
            goma_ctx->fem.fv_dot->em_hr[p] +=
                *goma_ctx->fem.esp_dot->em_hr[p][i] * goma_ctx->fem.bf[v]->phi[i];
          }
        }
      }
    }
  }

  if (pdgv[EM_H1_IMAG] || pdgv[EM_H2_IMAG] || pdgv[EM_H3_IMAG]) {
    for (p = 0; p < DIM; p++) {
      v = EM_H1_IMAG + p;
      goma_ctx->fem.fv->em_hi[p] = 0.0;
      goma_ctx->fem.fv_old->em_hi[p] = 0.0;
      goma_ctx->fem.fv_dot->em_hi[p] = 0.0;
      if (pdgv[v]) {
        dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
        for (i = 0; i < dofs; i++) {
          goma_ctx->fem.fv->em_hi[p] +=
              *goma_ctx->fem.esp->em_hi[p][i] * goma_ctx->fem.bf[v]->phi[i];

          if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
            goma_ctx->fem.fv_old->em_hi[p] +=
                *goma_ctx->fem.esp_old->em_hi[p][i] * goma_ctx->fem.bf[v]->phi[i];
            goma_ctx->fem.fv_dot->em_hi[p] +=
                *goma_ctx->fem.esp_dot->em_hi[p][i] * goma_ctx->fem.bf[v]->phi[i];
          }
        }
      }
    }
  }

  // internal wall distance calculations
  if (goma_ctx->fem.upd->turbulent_info->use_internal_wall_distance) {
    goma_ctx->fem.fv->wall_distance = 0.;
    if (pdgv[goma_ctx->fem.pd->ShapeVar]) {
      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[goma_ctx->fem.pd->ShapeVar]]
                 ->dof[goma_ctx->fem.pd->ShapeVar];
      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[goma_ctx->fem.pd->ShapeVar]]
                 ->dof[goma_ctx->fem.pd->ShapeVar];
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->wall_distance +=
            goma_ctx->fem.upd->turbulent_info->wall_distances
                [goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[goma_ctx->fem.pd->ShapeVar]]
                     ->gnn_list[goma_ctx->fem.pd->ShapeVar][i]] *
            goma_ctx->fem.bf[goma_ctx->fem.pd->ShapeVar]->phi[i];
      }
    }
  }

  goma_ctx->fem.fv->multi_contact_line_distance = 0;
  if (goma_ctx->fem.elc_glob[goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->mn]
          ->multi_contact_line_distances != NULL) {
    if (pdgv[goma_ctx->fem.pd->ShapeVar]) {
      dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[goma_ctx->fem.pd->ShapeVar];
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->multi_contact_line_distance +=
            goma_ctx->fem.elc_glob[goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->mn]
                ->multi_contact_line_distances[goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]
                                                   ->gnn_list[goma_ctx->fem.pd->ShapeVar][i]] *
            goma_ctx->fem.bf[goma_ctx->fem.pd->ShapeVar]->phi[i];
      }
    }
  }

  /*
   * External...
   */
  if (goma_ctx->fem.efv->ev) {
    int table_id = 0;
    v = EXTERNAL;
    for (w = 0; w < goma_ctx->fem.efv->Num_external_field; w++) {
      dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof_ext[w];
      goma_ctx->fem.fv->external_field[w] = 0.;
      goma_ctx->fem.fv_old->external_field[w] = 0.;
      goma_ctx->fem.fv_dot->external_field[w] = 0.;

      if (goma_ctx->fem.efv->i[w] != I_TABLE) {
        for (i = 0; i < dofs; i++) {
          goma_ctx->fem.fv->external_field[w] +=
              *goma_ctx->fem.evp->external_field[w][i] * goma_ctx->fem.bfex[w]->phi[i];

          if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
            goma_ctx->fem.fv_old->external_field[w] +=
                *goma_ctx->fem.evp->external_field[w][i] * goma_ctx->fem.bfex[w]->phi[i];
            goma_ctx->fem.fv_dot->external_field[w] += 0.;
          }
        }
      } else {
        double slope;
        goma_ctx->fem.fv->external_field[w] =
            interpolate_table(ext_Tables[table_id], goma_ctx->fem.fv->x, &slope, NULL);
        table_id++;
      }

      /*
       * If the variable name is velocity, and the momentum equations are not active,
       * load the external_fields into the velocity fv for use in Advection-diffusion analysis
       */
      if (strcmp(goma_ctx->fem.efv->name[w], "VX") == 0 &&
          !goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][VELOCITY1]) {
        goma_ctx->fem.fv->v[0] = goma_ctx->fem.fv->external_field[w];
        goma_ctx->fem.fv_old->v[0] = goma_ctx->fem.fv_old->external_field[w];
        goma_ctx->fem.fv_dot->v[0] = goma_ctx->fem.fv_dot->external_field[w];
      }
      if (strcmp(goma_ctx->fem.efv->name[w], "VY") == 0 &&
          !goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][VELOCITY2]) {
        goma_ctx->fem.fv->v[1] = goma_ctx->fem.fv->external_field[w];
        goma_ctx->fem.fv_old->v[1] = goma_ctx->fem.fv_old->external_field[w];
        goma_ctx->fem.fv_dot->v[1] = goma_ctx->fem.fv_dot->external_field[w];
      }
      if (strcmp(goma_ctx->fem.efv->name[w], "VZ") == 0 &&
          !goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][VELOCITY3]) {
        goma_ctx->fem.fv->v[2] = goma_ctx->fem.fv->external_field[w];
        goma_ctx->fem.fv_old->v[2] = goma_ctx->fem.fv_old->external_field[w];
        goma_ctx->fem.fv_dot->v[2] = goma_ctx->fem.fv_dot->external_field[w];
      }
      /*
       * If the variable name is mesh displacement, and the mesh
       * equations are not active, load the external_fields into
       * the mesh fv.  PRS NOTE: I don't think we will need these
       * for the decoupled JAS/GOMA, as the displacments are
       * swallowed into the mesh
       */
      if (strcmp(goma_ctx->fem.efv->name[w], "DMX") == 0 &&
          !goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][MESH_DISPLACEMENT1]) {
        goma_ctx->fem.fv->d[0] = goma_ctx->fem.fv->external_field[w];
        goma_ctx->fem.fv_old->d[0] = goma_ctx->fem.fv_old->external_field[w];
        goma_ctx->fem.fv_dot->d[0] = goma_ctx->fem.fv_dot->external_field[w];
      }
      if (strcmp(goma_ctx->fem.efv->name[w], "DMY") == 0 &&
          !goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][MESH_DISPLACEMENT2]) {
        goma_ctx->fem.fv->d[1] = goma_ctx->fem.fv->external_field[w];
        goma_ctx->fem.fv_old->d[1] = goma_ctx->fem.fv_old->external_field[w];
        goma_ctx->fem.fv_dot->d[1] = goma_ctx->fem.fv_dot->external_field[w];
      }
      if (strcmp(goma_ctx->fem.efv->name[w], "DMZ") == 0 &&
          !goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][MESH_DISPLACEMENT3]) {
        goma_ctx->fem.fv->d[2] = goma_ctx->fem.fv->external_field[w];
        goma_ctx->fem.fv_old->d[2] = goma_ctx->fem.fv_old->external_field[w];
        goma_ctx->fem.fv_dot->d[2] = goma_ctx->fem.fv_dot->external_field[w];
      }
      /*
       * If the variable name is porosity, and the porosity equation is not active,
       * load the external_fields into the porosity fv
       */
      if (strcmp(goma_ctx->fem.efv->name[w], "P_POR") == 0 &&
          !goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][POR_POROSITY]) {
        goma_ctx->fem.fv->porosity = goma_ctx->fem.fv->external_field[w];
        goma_ctx->fem.fv_old->porosity = goma_ctx->fem.fv_old->external_field[w];
        goma_ctx->fem.fv_dot->porosity = goma_ctx->fem.fv_dot->external_field[w];
      }
      if (strcmp(goma_ctx->fem.efv->name[w], "F") == 0 &&
          !goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][FILL]) {
        goma_ctx->fem.fv->F = goma_ctx->fem.fv->external_field[w];
        goma_ctx->fem.fv_old->F = goma_ctx->fem.fv_old->external_field[w];
        goma_ctx->fem.fv_dot->F = goma_ctx->fem.fv_dot->external_field[w];
      }
      if (strcmp(goma_ctx->fem.efv->name[w], "F1") == 0 &&
          !goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][PHASE1]) {
        goma_ctx->fem.fv->pF[0] = goma_ctx->fem.fv->external_field[w];
        goma_ctx->fem.fv_old->pF[0] = goma_ctx->fem.fv_old->external_field[w];
        goma_ctx->fem.fv_dot->pF[0] = goma_ctx->fem.fv_dot->external_field[w];
      }
    }
    if (!goma_ctx->fem.upd->turbulent_info->use_internal_wall_distance) {
      int i_d_wall = goma_ctx->fem.mp->dist_wall_ext_field_index;
      double d = goma_ctx->fem.fv->external_field[i_d_wall];
      goma_ctx->fem.fv->wall_distance = d;
    }
  } else {
    for (w = 0; w < goma_ctx->fem.efv->Num_external_field; w++) {
      goma_ctx->fem.fv->external_field[w] = 0.;
      goma_ctx->fem.fv_old->external_field[w] = 0.;
    }
  }

  /* initial displacements for TALE anneals. all for KINEMATIC DISPLACEMENT BC */
  if (goma_ctx->fem.efv->TALE) {
    for (w = 0; w < dim; w++) {
      v = MESH_DISPLACEMENT1 + w;
      dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
      goma_ctx->fem.fv->initial_displacements[w] = 0.;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->initial_displacements[w] +=
            *goma_ctx->fem.evp->initial_displacements[w][i] * goma_ctx->fem.bf[v]->phi[i];
      }
      v = SOLID_DISPLACEMENT1 + w;
      dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
      goma_ctx->fem.fv->initial_displacements[w + DIM] = 0.;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->initial_displacements[w + DIM] +=
            *goma_ctx->fem.evp->initial_displacements[w + DIM][i] * goma_ctx->fem.bf[v]->phi[i];
      }
    }
  } else {
    for (w = 0; w < dim; w++) {
      goma_ctx->fem.fv->initial_displacements[w] = 0.;
      goma_ctx->fem.fv->initial_displacements[w + DIM] = 0.;
    }
  }

  return (status);
}

int load_fv_vector(void)

/*******************************************************************************
 * load_fv() -- load up values of all relevant field variables at the
 *              current gauss pt
 *
 * input: (assume the appropriate parts of esp, bf, ei, pd are filled in)
 * ------
 *
 *
 * output: ( this routine fills in the following parts of the fv structure)
 * -------
 *		fv->
 *			T -- temperature	(scalar)
 *			v -- velocity		(vector)
 *			d -- mesh displacement	(vector)
 *			c -- concentration	(multiple scalars)
 *		      por -- porous media       (multiple scalars)
 *			P -- pressure		(scalar)
 *			S -- polymer stress	(tensor)
 *			G -- velocity gradient	(tensor)
 *                     pv -- particle velocity  (vector)
 *                     pG -- particle velocity gradient (tensor)
 *              mp->StateVector[] is filled in as well, for
 *                    pertinent entries that make up the specification of
 *                    the state of the material.
 *
 * NOTE: To accommodate shell elements, this function has been modified
 *       so that fv variables are not zeroed out when they are active
 *       on an element block other than the current one.
 *       The check done for variable v is then:
 *          if ( pd->v[pg->imtrx][v] || upd->vp[pg->imtrx][v] == -1 )
 *       In many cases below, this conditional zeroing is done in
 *       a separate small loop before the main one.
 *
 *
 * Return values:
 *		0 -- if things went OK
 *	       -1 -- if a problem occurred
 *
 * Created:	Fri Mar 18 06:44:41 MST 1994 pasacki@sandia.gov
 *
 * Modified:
 ***************************************************************************/
{
  int v; /* variable type indicator */
  int i; /* index */
  int p;
  int dofs; /* degrees of freedom for a var in the elem */
  int status = 0;
  int *pdgv = goma_ctx->fem.pd->gv;

  status = 0;

  /*
   * EM Wave Vectors...
   */
  if (pdgv[EM_E1_REAL] || pdgv[EM_E2_REAL] || pdgv[EM_E3_REAL]) {
    v = EM_E1_REAL;
    if (goma_ctx->fem.bf[v]->interpolation == I_N1) {
      for (p = 0; p < DIM; p++) {
        goma_ctx->fem.fv->em_er[p] = 0.0;
        goma_ctx->fem.fv_old->em_er[p] = 0.0;
        goma_ctx->fem.fv_dot->em_er[p] = 0.0;

        if (pdgv[v]) {
          dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->em_er[p] +=
                *goma_ctx->fem.esp->em_er[0][i] * goma_ctx->fem.bf[v]->phi_e[i][p];

            if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
              goma_ctx->fem.fv_old->em_er[p] +=
                  *goma_ctx->fem.esp_old->em_er[0][i] * goma_ctx->fem.bf[v]->phi_e[i][p];
              goma_ctx->fem.fv_dot->em_er[p] +=
                  *goma_ctx->fem.esp_dot->em_er[0][i] * goma_ctx->fem.bf[v]->phi_e[i][p];
            }
          }
        }
      }
    }
  }

  if (pdgv[EM_E1_IMAG] || pdgv[EM_E2_IMAG] || pdgv[EM_E3_IMAG]) {
    v = EM_E1_IMAG;
    if (goma_ctx->fem.bf[v]->interpolation == I_N1) {
      for (p = 0; p < DIM; p++) {
        goma_ctx->fem.fv->em_ei[p] = 0.0;
        goma_ctx->fem.fv_old->em_ei[p] = 0.0;
        goma_ctx->fem.fv_dot->em_ei[p] = 0.0;

        if (pdgv[v]) {
          dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->em_ei[p] +=
                *goma_ctx->fem.esp->em_ei[0][i] * goma_ctx->fem.bf[v]->phi_e[i][p];

            if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
              goma_ctx->fem.fv_old->em_ei[p] +=
                  *goma_ctx->fem.esp_old->em_ei[0][i] * goma_ctx->fem.bf[v]->phi_e[i][p];
              goma_ctx->fem.fv_dot->em_ei[p] +=
                  *goma_ctx->fem.esp_dot->em_ei[0][i] * goma_ctx->fem.bf[v]->phi_e[i][p];
            }
          }
        }
      }
    }
  }

  return (status);
}

/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/
int load_fv_grads(void)
/*******************************************************************************
 *
 * load_fv_grads() -- load relevant field variable gradients at this gauss pt
 *
 * input: (assume the appropriate parts of esp, bf, ei, pd are filled in)
 * ------
 *
 *
 * output: ( this routine fills in the following parts of the fv structure)
 * -------
 *		fv->
 *			grad_T -- temperature gradient
 *			grad_P -- pressure gradient
 *			grad_c -- species unknown gradient
 *			grad_F -- fill gradient
 *			grad_V -- voltage potential gradient
 *			div_v  -- divergence of velocity
 *			grad_v -- velocity gradient tensor
 *                      curl_v -- curl of velocity, a.k.a. vorticity
 *			div_d  -- divergence of displacement (dilatation)
 *			grad_d -- gradient of displacement ("strain")
 *                      grad_X -- gradient of the Eulerian solid reference state.
 *			grad_S -- gradient of the polymer stress tensor
 *			grad_G -- gradient of the velocity gradient tensor
 *                      grad_pv -- gradient of particle velocity
 *                      grad_p_liq, grad_p_gas, grad_porosity
 *                             -- gradient of porous media variables
 *                      grad_n -- gradient of level set normal vector
 *
 * NOTE: To accommodate shell elements, this function has been modified
 *       so that fv variables are not zeroed out when they are active
 *       on an element block other than the current one.
 *       The check done for variable v is then:
 *          if ( pd->v[pg->imtrx][v] || upd->vp[pg->imtrx][v] == -1 )
 *       In many cases below, this conditional zeroing is done in
 *       a separate small loop before the main one.
 *
 * Return values:
 *		0 -- if things went OK
 *	       -1 -- if a problem occurred
 *
 * Created:	Fri Mar 18 07:36:14 MST 1994 pasacki@sandia.gov
 *
 * Modified:	Tue Feb 21 11:08 MST 1995 pasacki@sandia.gov
 ***************************************************************************/
{
  int v;          /* variable type indicator */
  int i, a;       /* index */
  int p, q, r, s; /* dimension index */
  int dofs;       /* degrees of freedom */
  int w;          /* concentration species */
  int dim = goma_ctx->fem.pd->Num_Dim;
  int mode; /* modal counter */
  int status = 0;
  int transient_run = (goma_ctx->fem.pd->TimeIntegration != STEADY);
  BASIS_FUNCTIONS_STRUCT *bfn;

  /* Use a static flag so unused grads are zero on first call, but are not zero subsequently
   *  This is for efficieny
   */
  static int zero_unused_grads = FALSE;

  /*
   * grad(T)
   */
  if (goma_ctx->fem.pd->gv[TEMPERATURE]) {
    v = TEMPERATURE;
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_T[p] = 0.0;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_T[p] +=
            *goma_ctx->fem.esp->T[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][TEMPERATURE] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_T[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[FILM_HEIGHT]) {
    v = FILM_HEIGHT;
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_film_height[p] = 0.0;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_film_height[p] +=
            *goma_ctx->fem.esp->film_height[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][FILM_HEIGHT] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_film_height[p] = 0.0;
  }

  /*
   * grad(P)
   */

  if (goma_ctx->fem.pd->gv[PRESSURE]) {
    v = PRESSURE;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NO_UNROLL
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_P[p] = 0.0;

      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_P[p] +=
            *goma_ctx->fem.esp->P[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
#else
    grad_scalar_fv_fill(goma_ctx->fem.esp->P, goma_ctx->fem.bf[v]->grad_phi, dofs,
                        goma_ctx->fem.fv->grad_P);
    // if(transient_run && segregated)
    if (transient_run) {
      grad_scalar_fv_fill(goma_ctx->fem.esp_old->P, goma_ctx->fem.bf[v]->grad_phi, dofs,
                          goma_ctx->fem.fv_old->grad_P);
    }
#endif

  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][PRESSURE] == -1) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_P[p] = 0.0;
      // if(transient_run && segregated)
      if (transient_run) {
        goma_ctx->fem.fv_old->grad_P[p] = 0.0;
      }
    }
  }

  if (goma_ctx->fem.pd->gv[PSTAR]) {
    v = PSTAR;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NO_UNROLL
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_P[p] = 0.0;

      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_P[p] +=
            *goma_ctx->fem.esp->P[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
#else
    grad_scalar_fv_fill(goma_ctx->fem.esp->P_star, goma_ctx->fem.bf[v]->grad_phi, dofs,
                        goma_ctx->fem.fv->grad_P_star);
    // if(transient_run && segregated)
    if (transient_run) {
      grad_scalar_fv_fill(goma_ctx->fem.esp_old->P_star, goma_ctx->fem.bf[v]->grad_phi, dofs,
                          goma_ctx->fem.fv_old->grad_P_star);
    }
#endif

  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][PSTAR] == -1) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_P[p] = 0.0;
      // if(transient_run && segregated)
      if (transient_run) {
        goma_ctx->fem.fv_old->grad_P[p] = 0.0;
      }
    }
  }

  /*
   * grad(nn)
   */

  if (goma_ctx->fem.pd->gv[BOND_EVOLUTION]) {
    v = BOND_EVOLUTION;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_nn[p] = 0.;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_nn[p] +=
            *goma_ctx->fem.esp->nn[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  }

  /*
   * grad(F)
   */

  if (goma_ctx->fem.pd->gv[FILL]) {
    v = FILL;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NO_UNROLL
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_F[p] = 0.0;
      goma_ctx->fem.fv_old->grad_F[p] = 0.0;
      goma_ctx->fem.fv_dot->grad_F[p] = 0.0;

      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_F[p] +=
            *goma_ctx->fem.esp->F[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        if (transient_run) {
          /* keep this only for VOF/Taylor-Galerkin stuff */
          goma_ctx->fem.fv_old->grad_F[p] +=
              *goma_ctx->fem.esp_old->F[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];

          goma_ctx->fem.fv_dot->grad_F[p] +=
              *goma_ctx->fem.esp_dot->F[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        }
      }
    }
#else
    grad_scalar_fv_fill(goma_ctx->fem.esp->F, goma_ctx->fem.bf[v]->grad_phi, dofs,
                        goma_ctx->fem.fv->grad_F);

    if (transient_run) {
      grad_scalar_fv_fill(goma_ctx->fem.esp_old->F, goma_ctx->fem.bf[v]->grad_phi, dofs,
                          goma_ctx->fem.fv_old->grad_F);
      grad_scalar_fv_fill(goma_ctx->fem.esp_dot->F, goma_ctx->fem.bf[v]->grad_phi, dofs,
                          goma_ctx->fem.fv_dot->grad_F);
    }
#endif

  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][FILL] == -1) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_F[p] = 0.0;
      goma_ctx->fem.fv_old->grad_F[p] = 0.0;
      goma_ctx->fem.fv_dot->grad_F[p] = 0.0;
    }
  }

  /*
   * grad(H)
   */
  if (goma_ctx->fem.pd->gv[CURVATURE]) {
    v = CURVATURE;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_H[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_H[p] +=
            *goma_ctx->fem.esp->H[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][CURVATURE] == -1) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_H[p] = 0.0;
    }
  }

  /*
   * grad(V)
   */
  if (goma_ctx->fem.pd->gv[VOLTAGE]) {
    v = VOLTAGE;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_V[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_V[p] +=
            *goma_ctx->fem.esp->V[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][VOLTAGE] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_V[p] = 0.0;
  }

  /*
   * grad(Enorm)
   */
  if (goma_ctx->fem.pd->gv[ENORM]) {
    v = ENORM;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_Enorm[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_Enorm[p] +=
            *goma_ctx->fem.esp->Enorm[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  }

  /*
   * grad(qs)
   */
  if (goma_ctx->fem.pd->gv[SURF_CHARGE]) {
    v = SURF_CHARGE;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_qs[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        // HKM -> I changed dphidxi to grad_phi below. There didn't seem
        //        to be a case where the raw element derivative (dphidxi), which
        //        doesn't even take into account of the size of the element, should
        //        be used at this level.
        //   fv->grad_qs[p] += *esp->qs[i] * bf[v]->dphidxi[i][p];
        goma_ctx->fem.fv->grad_qs[p] +=
            *goma_ctx->fem.esp->qs[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SURF_CHARGE] == -1) {
    for (p = 0; p < VIM - 1; p++) {
      goma_ctx->fem.fv->grad_qs[p] = 0.0;
    }
  }

  /*
   * grad(SH)
   */

  if (goma_ctx->fem.pd->gv[SHEAR_RATE]) {
    v = SHEAR_RATE;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_SH[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_SH[p] +=
            *goma_ctx->fem.esp->SH[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHEAR_RATE] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_SH[p] = 0.0;
  }

  /*
   * grad(d)
   */

  if (goma_ctx->fem.pd->gv[MESH_DISPLACEMENT1] &&
      !ShellElementParentElementCoverageForVariable[MESH_DISPLACEMENT1]) {
    v = MESH_DISPLACEMENT1;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        goma_ctx->fem.fv->grad_d[p][q] = 0.0;
        goma_ctx->fem.fv_old->grad_d[p][q] = 0.0;
        for (r = 0; r < VIM; r++) {
          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->grad_d[p][q] +=
                *goma_ctx->fem.esp->d[r][i] * goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
            goma_ctx->fem.fv_old->grad_d[p][q] +=
                *goma_ctx->fem.esp_old->d[r][i] * goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
          }
        }
      }
    }
  } else {
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        goma_ctx->fem.fv->grad_d[p][q] = 0.0;
        goma_ctx->fem.fv_old->grad_d[p][q] = 0.0;
        goma_ctx->fem.fv->grad_d_dot[p][q] = 0.0;
      }
    }
  }

  /*
   * grad(d_dot)
   */
  if (goma_ctx->fem.pd->gv[MESH_DISPLACEMENT1] &&
      !ShellElementParentElementCoverageForVariable[MESH_DISPLACEMENT1]) {
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        v = MESH_DISPLACEMENT1 + p;
        if (goma_ctx->fem.pd->gv[v]) {
          dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
          goma_ctx->fem.fv->grad_d_dot[p][q] = 0.;
          for (r = 0; r < dim; r++) {
            for (i = 0; i < dofs; i++) {
              goma_ctx->fem.fv->grad_d_dot[p][q] +=
                  *goma_ctx->fem.esp_dot->d[r][i] * goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
            }
          }
        }
      }
    }
  }

  /*
   * grad(d_rs)
   */
  if (goma_ctx->fem.pd->gv[SOLID_DISPLACEMENT1]) {
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        v = SOLID_DISPLACEMENT1 + p;

        if (goma_ctx->fem.pd->gv[v]) {
          dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
          goma_ctx->fem.fv->grad_d_rs[p][q] = 0.0;
          for (r = 0; r < dim; r++) {
            for (i = 0; i < dofs; i++) {
              goma_ctx->fem.fv->grad_d_rs[p][q] +=
                  *goma_ctx->fem.esp->d_rs[r][i] * goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
            }
          }
        }
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SOLID_DISPLACEMENT1] == -1) {
    memset(goma_ctx->fem.fv->grad_d_rs, 0, sizeof(double) * VIM * VIM);
  }

  /*
   * div(d)
   */

  /*
   * div(d_dot)
   */

  if (goma_ctx->fem.pd->gv[MESH_DISPLACEMENT1] &&
      !ShellElementParentElementCoverageForVariable[MESH_DISPLACEMENT1]) {
    v = MESH_DISPLACEMENT1;
    goma_ctx->fem.fv->div_d_dot = 0.0;
    goma_ctx->fem.fv->div_d = 0.0;

    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->div_d_dot += goma_ctx->fem.fv->grad_d_dot[p][p];
      goma_ctx->fem.fv->div_d += goma_ctx->fem.fv->grad_d[p][p];
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][MESH_DISPLACEMENT1] == -1) {
    goma_ctx->fem.fv->div_d_dot = 0.0;
    goma_ctx->fem.fv->div_d = 0.0;
  }

  /*
   * div(d_rs)
   */

  if (goma_ctx->fem.pd->gv[SOLID_DISPLACEMENT1]) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->div_d_rs += goma_ctx->fem.fv->grad_d_rs[p][p];
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SOLID_DISPLACEMENT1] == -1) {
    goma_ctx->fem.fv->div_d_rs = 0.;
  }

  /////+(*
  ////+(* grad(v)
  ////+(*/
  if (goma_ctx->fem.pd->gv[VELOCITY1]) {
    v = VELOCITY1;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        goma_ctx->fem.fv->grad_v[p][q] = 0.0;
        goma_ctx->fem.fv_old->grad_v[p][q] = 0.0;
        for (r = 0; r < WIM; r++) {
          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->grad_v[p][q] +=
                (*goma_ctx->fem.esp->v[r][i]) * goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
            if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
              goma_ctx->fem.fv_old->grad_v[p][q] +=
                  (*goma_ctx->fem.esp_old->v[r][i]) * goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
            }
          }
        }
      }
    }
  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][VELOCITY1] == -1) {
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        goma_ctx->fem.fv->grad_v[p][q] = 0.0;

        // if(segregated)
        goma_ctx->fem.fv_old->grad_v[p][q] = 0.0;
      }
    }
  }

  if (goma_ctx->fem.pd->gv[USTAR]) {
    v = USTAR;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        goma_ctx->fem.fv->grad_v_star[p][q] = 0.0;
        for (r = 0; r < WIM; r++) {
          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->grad_v_star[p][q] +=
                (*goma_ctx->fem.esp->v_star[r][i]) * goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
          }
        }
      }
    }
  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][USTAR] == -1) {
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        goma_ctx->fem.fv->grad_v_star[p][q] = 0.0;
      }
    }
  }

  /*
   * grad(pv), particle velocity gradients.
   */
  if (goma_ctx->fem.pd->gv[PVELOCITY1]) {
    v = PVELOCITY1;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        for (r = 0; r < WIM; r++) {
          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->grad_pv[p][q] +=
                *goma_ctx->fem.esp->pv[r][i] * goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
          }
        }
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][PVELOCITY1] == -1) {
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        goma_ctx->fem.fv->grad_pv[p][q] = 0.0;
      }
    }
  }

  /*
   * grad(ext_v), extension velocity gradients.
   */

  if (goma_ctx->fem.pd->gv[EXT_VELOCITY]) {
    v = EXT_VELOCITY;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_ext_v[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_ext_v[p] +=
            *goma_ctx->fem.esp->ext_v[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EXT_VELOCITY] == -1) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_ext_v[p] = 0.0;
    }
  }

  /*
   *  grad( n ) .  Level set or shell normal vector gradient
   */
  if (goma_ctx->fem.pd->gv[NORMAL1] || goma_ctx->fem.pd->gv[SHELL_NORMAL1]) {
    if (goma_ctx->fem.pd->gv[NORMAL1])
      v = NORMAL1;
    if (goma_ctx->fem.pd->gv[SHELL_NORMAL1])
      v = SHELL_NORMAL1;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        goma_ctx->fem.fv->grad_n[p][q] = 0.0;
        goma_ctx->fem.fv_old->grad_n[p][q] = 0.0;
        for (r = 0; r < dim; r++) {
          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->grad_n[p][q] +=
                *(goma_ctx->fem.esp->n[r][i]) * goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
            goma_ctx->fem.fv_old->grad_n[p][q] +=
                *(goma_ctx->fem.esp_old->n[r][i]) * goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
          }
        }
      }
    }
  } else if ((zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][NORMAL1] == -1) ||
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_NORMAL1] == -1) {
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        goma_ctx->fem.fv->grad_n[p][q] = 0.0;
        goma_ctx->fem.fv_old->grad_n[p][q] = 0.0;
      }
    }
  }

  /*
   *  div(n)
   */
  if (goma_ctx->fem.pd->gv[NORMAL1] || goma_ctx->fem.pd->gv[SHELL_NORMAL1]) {
    goma_ctx->fem.fv->div_n = 0.0;
    goma_ctx->fem.fv_old->div_n = 0.0;
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->div_n += goma_ctx->fem.fv->grad_n[p][p];
      goma_ctx->fem.fv_old->div_n += goma_ctx->fem.fv_old->grad_n[p][p];
    }
  }

  /*
   * div_s(n)
   */
  if (goma_ctx->fem.pd->gv[NORMAL1] || goma_ctx->fem.pd->gv[SHELL_NORMAL1]) {
    goma_ctx->fem.fv->div_s_n = 0.0;
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        goma_ctx->fem.fv->div_s_n +=
            (delta(p, q) * goma_ctx->fem.fv->grad_n[p][q] -
             goma_ctx->fem.fv->n[p] * goma_ctx->fem.fv->n[q] * goma_ctx->fem.fv->grad_n[q][p]);
      }
    }
  }

  // Calculation of the surface curvature dyadic
  //             = - del_s (n)  = - (I - n n ) grad n
  if (goma_ctx->fem.pd->gv[SHELL_NORMAL1]) {
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        goma_ctx->fem.fv->surfCurvatureDyadic[p][q] = -goma_ctx->fem.fv->grad_n[p][q];
        for (a = 0; a < VIM; a++) {
          goma_ctx->fem.fv->surfCurvatureDyadic[p][q] +=
              goma_ctx->fem.fv->n[p] * goma_ctx->fem.fv->n[a] * goma_ctx->fem.fv->grad_n[a][q];
        }
      }
    }
  }

  /*
   * grad(E_field), Electric field gradients.
   */
  if (goma_ctx->fem.pd->gv[EFIELD1]) {
    v = EFIELD1;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    v = EFIELD1;

    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        goma_ctx->fem.fv->grad_E_field[p][q] = 0.0;
        for (r = 0; r < WIM; r++) {
          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->grad_E_field[p][q] +=
                *goma_ctx->fem.esp->E_field[r][i] * goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
          }
        }
      }
    }
  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EFIELD1] != -1) {
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        goma_ctx->fem.fv->grad_E_field[p][q] = 0.0;
      }
    }
  }

  /* Phase Functions */

  if (goma_ctx->fem.pd->gv[PHASE1]) {
    for (r = 0; r < goma_ctx->fem.pfd->num_phase_funcs; r++) {
      v = PHASE1 + r;
      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

      for (p = 0; p < VIM; p++) {
        goma_ctx->fem.fv->grad_pF[r][p] = 0.0;

        for (i = 0; i < dofs; i++) {
          goma_ctx->fem.fv->grad_pF[r][p] +=
              *goma_ctx->fem.esp->pF[r][i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        }
      }
    }
  }

  /*
   * curl(v)
   */
  if (CURL_V != -1 && !InShellElementWithParentElementCoverage) {
    v = VELOCITY1;
    bfn = goma_ctx->fem.bf[v];

    if (goma_ctx->fem.pd->gv[v] || goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][v] != -1) {
      for (p = 0; p < DIM; p++)
        goma_ctx->fem.fv->curl_v[p] = 0.0;
    }

    if (goma_ctx->fem.pd->gv[v]) {
      /* Always compute all three components of the vorticity
       * vector.  If we are really in 2D, then the output routine will
       * only output the 3rd component. DIM is equal to 3.
       */
      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[VELOCITY1];
      for (p = 0; p < DIM; p++) {
        for (i = 0; i < dofs; i++) {
          for (a = 0; a < VIM; a++) /* VIM */
          {
            goma_ctx->fem.fv->curl_v[p] += *goma_ctx->fem.esp->v[a][i] * bfn->curl_phi_e[i][a][p];
          }
        }
      }
    }
  }

  if (goma_ctx->fem.pd->gv[EM_E1_REAL]) {
    v = EM_E1_REAL;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    bfn = goma_ctx->fem.bf[v];

    if (bfn->interpolation == I_N1) {
      for (p = 0; p < DIM; p++) {
        goma_ctx->fem.fv->curl_em_er[p] = 0.0;
        for (i = 0; i < dofs; i++) {
          goma_ctx->fem.fv->curl_em_er[p] += *goma_ctx->fem.esp->em_er[0][i] * bfn->curl_phi[i][p];
        }
      }
    } else {
      for (p = 0; p < DIM; p++) {
        goma_ctx->fem.fv->curl_em_er[p] = 0.0;
        for (i = 0; i < dofs; i++) {
          for (a = 0; a < dim; a++) {
            goma_ctx->fem.fv->curl_em_er[p] +=
                *goma_ctx->fem.esp->em_er[a][i] * bfn->curl_phi_e[i][a][p];
          }
        }
      }
    }
  }

  if (goma_ctx->fem.pd->gv[EM_E1_IMAG]) {
    v = EM_E1_IMAG;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    bfn = goma_ctx->fem.bf[v];
    if (bfn->interpolation == I_N1) {
      for (p = 0; p < DIM; p++) {
        goma_ctx->fem.fv->curl_em_ei[p] = 0.0;
        for (i = 0; i < dofs; i++) {
          goma_ctx->fem.fv->curl_em_ei[p] += *goma_ctx->fem.esp->em_ei[0][i] * bfn->curl_phi[i][p];
        }
      }
    } else {
      for (p = 0; p < DIM; p++) {
        goma_ctx->fem.fv->curl_em_ei[p] = 0.0;
        for (i = 0; i < dofs; i++) {
          for (a = 0; a < dim; a++) {
            goma_ctx->fem.fv->curl_em_ei[p] +=
                *goma_ctx->fem.esp->em_ei[a][i] * bfn->curl_phi_e[i][a][p];
          }
        }
      }
    }
  }

  /*
   * div(v)
   */

  if (goma_ctx->fem.pd->gv[VELOCITY1]) {
    goma_ctx->fem.fv->div_v = 0.0;
    if (VIM == 3 && goma_ctx->fem.pd->CoordinateSystem != CARTESIAN_2pt5D)
      goma_ctx->fem.fv_old->div_v = 0.0;
    for (a = 0; a < VIM; a++) {
      goma_ctx->fem.fv->div_v += goma_ctx->fem.fv->grad_v[a][a];
      // if(segregated)
      goma_ctx->fem.fv_old->div_v += goma_ctx->fem.fv_old->grad_v[a][a];
    }
  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][VELOCITY1] == -1) {
    goma_ctx->fem.fv->div_v = 0.0;
    goma_ctx->fem.fv_old->div_v = 0.0;
  }

  if (goma_ctx->fem.pd->gv[USTAR]) {
    goma_ctx->fem.fv->div_v_star = 0.0;
    for (a = 0; a < VIM; a++) {
      goma_ctx->fem.fv->div_v_star += goma_ctx->fem.fv->grad_v_star[a][a];
    }
  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][USTAR] == -1) {
    goma_ctx->fem.fv->div_v_star = 0.0;
  }

  /*
   * div(pv)
   */

  if (goma_ctx->fem.pd->gv[PVELOCITY1]) {
    v = PVELOCITY1;
    goma_ctx->fem.fv->div_pv = 0.0;
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->div_pv += goma_ctx->fem.fv->grad_pv[p][p];
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][PVELOCITY1] == -1) {
    goma_ctx->fem.fv->div_pv = 0.0;
  }

  /*
   * Species unknown ...
   */

  if (goma_ctx->fem.pd->gv[MASS_FRACTION]) {
    for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
      v = MASS_FRACTION;
      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
      for (p = 0; p < VIM; p++) {
        goma_ctx->fem.fv->grad_c[w][p] = 0.;
        goma_ctx->fem.fv_old->grad_c[w][p] = 0.;

        for (i = 0; i < dofs; i++) {
          goma_ctx->fem.fv->grad_c[w][p] +=
              *goma_ctx->fem.esp->c[w][i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
          if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
            /* keep this only for VOF/Taylor-Galerkin stuff */
            goma_ctx->fem.fv_old->grad_c[w][p] +=
                *goma_ctx->fem.esp_old->c[w][i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
          }
        }
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][MASS_FRACTION] == -1) {
    for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
      for (p = 0; p < VIM; p++) {
        goma_ctx->fem.fv->grad_c[w][p] = 0.;
        goma_ctx->fem.fv_old->grad_c[w][p] = 0.;
      }
    }
  }

  /*
   * Porous media unknowns ...
   */

  if (goma_ctx->fem.pd->gv[POR_LIQ_PRES]) {
    v = POR_LIQ_PRES;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_p_liq[p] = 0.0;
      goma_ctx->fem.fv_old->grad_p_liq[p] = 0.0;

      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_p_liq[p] +=
            *goma_ctx->fem.esp->p_liq[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
          /* keep this only for VOF/Taylor-Galerkin stuff */
          goma_ctx->fem.fv_old->grad_p_liq[p] +=
              *goma_ctx->fem.esp_old->p_liq[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        }
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][POR_LIQ_PRES] == -1) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_p_liq[p] = 0.0;
      goma_ctx->fem.fv_old->grad_p_liq[p] = 0.0;
    }
  }

  if (goma_ctx->fem.pd->gv[POR_GAS_PRES]) {
    v = POR_GAS_PRES;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_p_gas[p] = 0.0;
      goma_ctx->fem.fv_old->grad_p_gas[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_p_gas[p] +=
            *goma_ctx->fem.esp->p_gas[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
          /* keep this only for VOF/Taylor-Galerkin stuff */
          goma_ctx->fem.fv_old->grad_p_gas[p] +=
              *goma_ctx->fem.esp_old->p_gas[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        }
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][POR_GAS_PRES] == -1) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_p_gas[p] = 0.0;
      goma_ctx->fem.fv_old->grad_p_gas[p] = 0.0;
    }
  }

  if (goma_ctx->fem.pd->gv[POR_POROSITY]) {
    v = POR_POROSITY;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_porosity[p] = 0.0;
      goma_ctx->fem.fv_old->grad_porosity[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_porosity[p] +=
            *goma_ctx->fem.esp->porosity[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
          /* keep this only for VOF/Taylor-Galerkin stuff */
          goma_ctx->fem.fv_old->grad_porosity[p] +=
              *goma_ctx->fem.esp_old->porosity[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        }
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][POR_POROSITY] == -1) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_porosity[p] = 0.0;
      goma_ctx->fem.fv_old->grad_porosity[p] = 0.0;
    }
  }

  if (goma_ctx->fem.pd->gv[POR_TEMP]) {
    v = POR_TEMP;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_T[p] = 0.;
      goma_ctx->fem.fv_old->grad_T[p] = 0.;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_T[p] +=
            *goma_ctx->fem.esp->T[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
          /* keep this only for VOF/Taylor-Galerkin stuff */
          goma_ctx->fem.fv_old->grad_T[p] +=
              *goma_ctx->fem.esp_old->T[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        }
      }
    }
  }

  /*
   * grad(S), mode 0
   */

  if (goma_ctx->fem.pd->gv[POLYMER_STRESS11]) {
    v = POLYMER_STRESS11;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (mode = 0; mode < goma_ctx->fem.vn->modes; mode++) {
      for (p = 0; p < VIM; p++) {
        for (q = 0; q < VIM; q++) {
          for (r = 0; r < VIM; r++) {
            goma_ctx->fem.fv->grad_S[mode][r][p][q] = 0.;

            for (i = 0; i < dofs; i++) {
              if (p <= q) {
                goma_ctx->fem.fv->grad_S[mode][r][p][q] +=
                    *goma_ctx->fem.esp->S[mode][p][q][i] * goma_ctx->fem.bf[v]->grad_phi[i][r];
              } else {
                goma_ctx->fem.fv->grad_S[mode][r][p][q] +=
                    *goma_ctx->fem.esp->S[mode][q][p][i] * goma_ctx->fem.bf[v]->grad_phi[i][r];
              }
            }
          }
        }
      }
    }

    /*
     * div(S) - this is a vector!
     */
    for (mode = 0; mode < goma_ctx->fem.vn->modes; mode++) {
      for (r = 0; r < dim; r++) {
        goma_ctx->fem.fv->div_S[mode][r] = 0.0;

        for (q = 0; q < dim; q++) {
          goma_ctx->fem.fv->div_S[mode][r] += goma_ctx->fem.fv->grad_S[mode][q][q][r];
        }
      }
    }

    if (goma_ctx->fem.pd->CoordinateSystem != CARTESIAN) {
      for (mode = 0; mode < goma_ctx->fem.vn->modes; mode++) {
        for (s = 0; s < VIM; s++) {
          for (r = 0; r < VIM; r++) {
            for (p = 0; p < VIM; p++) {
              goma_ctx->fem.fv->div_S[mode][s] +=
                  goma_ctx->fem.fv->S[mode][p][s] * goma_ctx->fem.fv->grad_e[p][r][s];
            }
          }
        }
        for (s = 0; s < VIM; s++) {
          for (r = 0; r < VIM; r++) {
            for (q = 0; q < VIM; q++) {
              goma_ctx->fem.fv->div_S[mode][s] +=
                  goma_ctx->fem.fv->S[mode][r][q] * goma_ctx->fem.fv->grad_e[q][r][s];
            }
          }
        }
      }
    }

  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][POLYMER_STRESS11] == -1) {
    for (mode = 0; mode < goma_ctx->fem.vn->modes; mode++) {
      for (p = 0; p < VIM; p++) {
        goma_ctx->fem.fv->div_S[mode][p] = 0.0;
        for (q = 0; q < VIM; q++) {
          for (r = 0; r < VIM; r++) {
            goma_ctx->fem.fv->grad_S[mode][r][p][q] = 0.;
          }
        }
      }
    }
  }

  /*
   * grad(G)
   */

  if (goma_ctx->fem.pd->gv[VELOCITY_GRADIENT11]) {
    v = VELOCITY_GRADIENT11;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        for (r = 0; r < VIM; r++) {
          goma_ctx->fem.fv->grad_G[r][p][q] = 0.0;
          goma_ctx->fem.fv->grad_Gt[r][p][q] = 0.0;

          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->grad_G[r][p][q] +=
                *goma_ctx->fem.esp->G[p][q][i] * goma_ctx->fem.bf[v]->grad_phi[i][r];
            goma_ctx->fem.fv->grad_Gt[r][p][q] +=
                *goma_ctx->fem.esp->G[q][p][i] * goma_ctx->fem.bf[v]->grad_phi[i][r];
          }
        }
      }
    }

    /*
     * div(G) - this is a vector!
     */
    for (r = 0; r < dim; r++) {
      goma_ctx->fem.fv->div_G[r] = 0.0;
      for (q = 0; q < dim; q++) {
        goma_ctx->fem.fv->div_G[r] += goma_ctx->fem.fv->grad_G[q][q][r];
      }
    }

    if (goma_ctx->fem.pd->CoordinateSystem != CARTESIAN) {
      for (s = 0; s < VIM; s++) {
        for (r = 0; r < VIM; r++) {
          for (p = 0; p < VIM; p++) {
            goma_ctx->fem.fv->div_G[s] +=
                goma_ctx->fem.fv->G[p][s] * goma_ctx->fem.fv->grad_e[p][r][s];
          }
        }
      }

      for (s = 0; s < VIM; s++) {
        for (r = 0; r < VIM; r++) {
          for (q = 0; q < VIM; q++) {
            goma_ctx->fem.fv->div_G[s] +=
                goma_ctx->fem.fv->G[r][q] * goma_ctx->fem.fv->grad_e[q][r][s];
          }
        }
      }
    }

    /*
     * div(Gt) - this is a vector and the divergence of the
     * transpose of G.
     */
    for (r = 0; r < dim; r++) {
      goma_ctx->fem.fv->div_Gt[r] = 0.0;
      for (q = 0; q < dim; q++) {
        goma_ctx->fem.fv->div_Gt[r] += goma_ctx->fem.fv->grad_G[q][r][q];
      }
    }

    if (goma_ctx->fem.pd->CoordinateSystem != CARTESIAN) {
      for (s = 0; s < VIM; s++) {
        for (r = 0; r < VIM; r++) {
          for (p = 0; p < VIM; p++) {
            goma_ctx->fem.fv->div_Gt[s] +=
                goma_ctx->fem.fv->G[s][p] * goma_ctx->fem.fv->grad_e[p][r][s];
          }
        }
      }

      for (s = 0; s < VIM; s++) {
        for (r = 0; r < VIM; r++) {
          for (q = 0; q < VIM; q++) {
            goma_ctx->fem.fv->div_Gt[s] +=
                goma_ctx->fem.fv->G[q][r] * goma_ctx->fem.fv->grad_e[q][r][s];
          }
        }
      }
    }

  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][VELOCITY_GRADIENT11] == -1) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->div_G[p] = 0.0;
      goma_ctx->fem.fv->div_Gt[p] = 0.0;
      for (q = 0; q < VIM; q++) {
        for (r = 0; r < VIM; r++) {
          goma_ctx->fem.fv->grad_G[r][p][q] = 0.;
        }
      }
    }
  }

  /*
   * grad(vd)
   */
  if (goma_ctx->fem.pd->gv[VORT_DIR1]) {
    v = VORT_DIR1;
    ;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

    for (p = 0; p < DIM; p++) {
      for (q = 0; q < DIM; q++) {
        goma_ctx->fem.fv->grad_vd[p][q] = 0.0;
        for (r = 0; r < DIM; r++) {
          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->grad_vd[p][q] +=
                *goma_ctx->fem.esp->vd[r][i] * goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
          }
        }
      }
    }
  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][VORT_DIR1] == -1) {
    for (p = 0; p < DIM; p++) {
      for (q = 0; q < DIM; q++) {
        goma_ctx->fem.fv->grad_vd[p][q] = 0.0;
      }
    }
  }
  /*
   * div(vd)
   */

  if (goma_ctx->fem.pd->gv[VORT_DIR1]) {
    goma_ctx->fem.fv->div_vd = 0.0;
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->div_vd += goma_ctx->fem.fv->grad_vd[p][p];
    }
  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][VORT_DIR1] == -1)
    goma_ctx->fem.fv->div_vd = 0.0;

  /*
   * grad_sh_K
   * Gradient of curvature in structural shell
   */

  if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][SHELL_CURVATURE]) {
    v = SHELL_CURVATURE;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_sh_K[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_sh_K[p] +=
            *goma_ctx->fem.esp->sh_K[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_CURVATURE] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_sh_K[p] = 0.0;
  }

  if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][SHELL_CURVATURE2]) {
    v = SHELL_CURVATURE2;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_sh_K2[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_sh_K2[p] +=
            *goma_ctx->fem.esp->sh_K2[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_CURVATURE2] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_sh_K2[p] = 0.0;
  }

  /*
   * grad(n_dot_curl_s_v)
   *   This is the normal gradient of a scalar field defined on a shell.
   */
  if (goma_ctx->fem.pd->gv[N_DOT_CURL_V]) {
    v = N_DOT_CURL_V;
    bfn = goma_ctx->fem.bf[v];
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_n_dot_curl_s_v[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_n_dot_curl_s_v[p] +=
            *goma_ctx->fem.esp->n_dot_curl_s_v[i] * bfn->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][N_DOT_CURL_V] == -1) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_n_dot_curl_s_v[p] = 0.0;
    }
  }

  /*
   * grad(div_s_v)
   *   This is the normal gradient of a scalar field defined on a shell.
   *        The scalar field is the surface divergence of the velocity
   */
  if (goma_ctx->fem.pd->gv[SHELL_SURF_DIV_V]) {
    v = SHELL_SURF_DIV_V;
    bfn = goma_ctx->fem.bf[v];
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_div_s_v[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_div_s_v[p] += *goma_ctx->fem.esp->div_s_v[i] * bfn->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_SURF_DIV_V] == -1) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_div_s_v[p] = 0.0;
    }
  }

  /*
   * grad(curv)
   *   This is the normal gradient of a scalar field defined on a shell.
   *        The scalar field is the mean curvature of the surface
   */
  if (goma_ctx->fem.pd->gv[SHELL_SURF_CURV]) {
    v = SHELL_SURF_CURV;
    bfn = goma_ctx->fem.bf[v];
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_curv[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_curv[p] += *goma_ctx->fem.esp->curv[i] * bfn->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_SURF_CURV] == -1) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_curv[p] = 0.0;
    }
  }

  /*
   * grad(grad_s_v_dot_n)
   *   This is the normal gradient of a scalar field defined on a shell.
   *        The scalar field is the vector component of the grad_s_v_dot_n
   */
  if (goma_ctx->fem.pd->gv[GRAD_S_V_DOT_N1]) {
    for (r = 0; r < dim; r++) {
      v = GRAD_S_V_DOT_N1 + r;
      bfn = goma_ctx->fem.bf[v];
      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
      for (p = 0; p < VIM; p++) {
        goma_ctx->fem.fv->serialgrad_grad_s_v_dot_n[r][p] = 0.0;
        for (i = 0; i < dofs; i++) {
          goma_ctx->fem.fv->serialgrad_grad_s_v_dot_n[r][p] +=
              *(goma_ctx->fem.esp->grad_v_dot_n[r][i]) * bfn->grad_phi[i][p];
        }
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][GRAD_S_V_DOT_N1] == -1) {
    for (r = 0; r < dim; r++) {
      for (p = 0; p < VIM; p++) {
        goma_ctx->fem.fv->serialgrad_grad_s_v_dot_n[r][p] = 0.0;
      }
    }
  }

  /*
   * grad(sh_J)
   */
  if (goma_ctx->fem.pd->gv[SHELL_DIFF_FLUX]) {
    v = SHELL_DIFF_FLUX;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_sh_J[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        // HKM -> I changed dphidxi to grad_phi below. There didn't seem
        //        to be a case where the raw element derivative (dphidxi) which
        //        doesn't even take into account of the size of the element should
        //        be used at this level.
        //      fv->grad_sh_J[p] += *esp->sh_J[i] * bf[v]->dphidxi[i][p];
        goma_ctx->fem.fv->grad_sh_J[p] +=
            *goma_ctx->fem.esp->sh_J[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_DIFF_FLUX] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_sh_J[p] = 0.0;
  }

  /* grad(EDDY_NU)
   *
   */
  if (goma_ctx->fem.pd->gv[EDDY_NU]) {
    v = EDDY_NU;
    bfn = goma_ctx->fem.bf[v];
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_eddy_nu[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_eddy_nu[p] += *goma_ctx->fem.esp->eddy_nu[i] * bfn->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EDDY_NU] == -1) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_eddy_nu[p] = 0.0;
    }
  }

  if (goma_ctx->fem.pd->gv[TURB_K]) {
    v = TURB_K;
    bfn = goma_ctx->fem.bf[v];
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_turb_k[p] = 0.0;
      goma_ctx->fem.fv_old->grad_turb_k[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_turb_k[p] += *goma_ctx->fem.esp->turb_k[i] * bfn->grad_phi[i][p];
        goma_ctx->fem.fv_old->grad_turb_k[p] +=
            *goma_ctx->fem.esp_old->turb_k[i] * bfn->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][TURB_K] == -1) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_turb_k[p] = 0.0;
    }
  }

  if (goma_ctx->fem.pd->gv[TURB_OMEGA]) {
    v = TURB_OMEGA;
    bfn = goma_ctx->fem.bf[v];
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_turb_omega[p] = 0.0;
      goma_ctx->fem.fv_old->grad_turb_omega[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_turb_omega[p] +=
            *goma_ctx->fem.esp->turb_omega[i] * bfn->grad_phi[i][p];
        goma_ctx->fem.fv_old->grad_turb_omega[p] +=
            *goma_ctx->fem.esp_old->turb_omega[i] * bfn->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][TURB_OMEGA] == -1) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_turb_omega[p] = 0.0;
    }
  }

  /*
   * grad(APR)
   */

  if (goma_ctx->fem.pd->gv[ACOUS_PREAL]) {
    v = ACOUS_PREAL;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_apr[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_apr[p] +=
            *goma_ctx->fem.esp->apr[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][ACOUS_PREAL] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_apr[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[ACOUS_PIMAG]) {
    v = ACOUS_PIMAG;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_api[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_api[p] +=
            *goma_ctx->fem.esp->api[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][ACOUS_PIMAG] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_api[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[ACOUS_REYN_STRESS]) {
    v = ACOUS_REYN_STRESS;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_ars[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_ars[p] +=
            *goma_ctx->fem.esp->ars[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][ACOUS_REYN_STRESS] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_ars[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[SHELL_BDYVELO]) {
    v = SHELL_BDYVELO;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_sh_bv[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_sh_bv[p] +=
            *goma_ctx->fem.esp->sh_bv[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_BDYVELO] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_sh_bv[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[SHELL_LUBP]) {
    v = SHELL_LUBP;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_sh_p[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_sh_p[p] +=
            *goma_ctx->fem.esp->sh_p[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_LUBP] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_sh_p[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[LUBP]) {
    v = LUBP;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_lubp[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_lubp[p] +=
            *goma_ctx->fem.esp->lubp[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        goma_ctx->fem.fv_old->grad_lubp[p] +=
            *goma_ctx->fem.esp_old->lubp[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][LUBP] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_lubp[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[LUBP_2]) {
    v = LUBP_2;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_lubp_2[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_lubp_2[p] +=
            *goma_ctx->fem.esp->lubp_2[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        goma_ctx->fem.fv_old->grad_lubp_2[p] +=
            *goma_ctx->fem.esp_old->lubp_2[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][LUBP_2] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_lubp_2[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[SHELL_LUB_CURV]) {
    v = SHELL_LUB_CURV;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_sh_l_curv[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_sh_l_curv[p] +=
            *goma_ctx->fem.esp->sh_l_curv[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_LUB_CURV] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_sh_l_curv[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[SHELL_LUB_CURV_2]) {
    v = SHELL_LUB_CURV_2;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_sh_l_curv_2[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_sh_l_curv_2[p] +=
            *goma_ctx->fem.esp->sh_l_curv_2[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_LUB_CURV_2] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_sh_l_curv_2[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[SHELL_PRESS_OPEN]) {
    v = SHELL_PRESS_OPEN;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_sh_p_open[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_sh_p_open[p] +=
            *goma_ctx->fem.esp->sh_p_open[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        goma_ctx->fem.fv_old->grad_sh_p_open[p] +=
            *goma_ctx->fem.esp_old->sh_p_open[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_PRESS_OPEN] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_sh_p_open[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[SHELL_PRESS_OPEN_2]) {
    v = SHELL_PRESS_OPEN_2;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_sh_p_open_2[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_sh_p_open_2[p] +=
            *goma_ctx->fem.esp->sh_p_open_2[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        goma_ctx->fem.fv_old->grad_sh_p_open_2[p] +=
            *goma_ctx->fem.esp_old->sh_p_open_2[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_PRESS_OPEN_2] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_sh_p_open_2[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[SHELL_SAT_1]) {
    v = SHELL_SAT_1;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_sh_sat_1[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_sh_sat_1[p] +=
            *goma_ctx->fem.esp->sh_sat_1[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_SAT_1] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_sh_sat_1[p] = 0.0;
  }
  if (goma_ctx->fem.pd->gv[SHELL_FILMP]) {
    v = SHELL_FILMP;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_sh_fp[p] = 0.0;
      goma_ctx->fem.fv_old->grad_sh_fp[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_sh_fp[p] +=
            *goma_ctx->fem.esp->sh_fp[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        goma_ctx->fem.fv_old->grad_sh_fp[p] +=
            *goma_ctx->fem.esp_old->sh_fp[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_FILMP] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_sh_fp[p] = goma_ctx->fem.fv_old->grad_sh_fp[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[SHELL_FILMH]) {
    v = SHELL_FILMH;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_sh_fh[p] = 0.0;
      goma_ctx->fem.fv_old->grad_sh_fh[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_sh_fh[p] +=
            *goma_ctx->fem.esp->sh_fh[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        goma_ctx->fem.fv_old->grad_sh_fh[p] +=
            *goma_ctx->fem.esp_old->sh_fh[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_FILMH] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_sh_fh[p] = goma_ctx->fem.fv_old->grad_sh_fh[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[SHELL_PARTC]) {
    v = SHELL_PARTC;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_sh_pc[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_sh_pc[p] +=
            *goma_ctx->fem.esp->sh_pc[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_PARTC] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_sh_pc[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[SHELL_TEMPERATURE]) {
    v = SHELL_TEMPERATURE;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_sh_t[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_sh_t[p] +=
            *goma_ctx->fem.esp->sh_t[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_TEMPERATURE] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_sh_t[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[LIGHT_INTP]) {
    v = LIGHT_INTP;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_poynt[0][p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_poynt[0][p] +=
            *goma_ctx->fem.esp->poynt[0][i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][LIGHT_INTP] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_poynt[0][p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[LIGHT_INTM]) {
    v = LIGHT_INTM;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_poynt[1][p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_poynt[1][p] +=
            *goma_ctx->fem.esp->poynt[1][i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][LIGHT_INTM] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_poynt[1][p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[LIGHT_INTD]) {
    v = LIGHT_INTD;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_poynt[2][p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_poynt[2][p] +=
            *goma_ctx->fem.esp->poynt[2][i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][LIGHT_INTD] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_poynt[2][p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[MOMENT0]) {
    for (r = 0; r < MAX_MOMENTS; r++) {
      v = MOMENT0 + r;
      dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];

      for (p = 0; p < VIM; p++) {
        goma_ctx->fem.fv->grad_moment[r][p] = 0.0;
        goma_ctx->fem.fv_old->grad_moment[r][p] = 0.0;

        for (i = 0; i < dofs; i++) {
          goma_ctx->fem.fv->grad_moment[r][p] +=
              *goma_ctx->fem.esp->moment[r][i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
          goma_ctx->fem.fv_old->grad_moment[r][p] +=
              *goma_ctx->fem.esp_old->moment[r][i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        }
      }
    }
  }

  if (goma_ctx->fem.pd->gv[DENSITY_EQN]) {
    v = DENSITY_EQN;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
#ifdef DO_NO_UNROLL
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_rho[p] = 0.0;

      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_rho[p] +=
            *goma_ctx->fem.esp->rho[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
#else
    grad_scalar_fv_fill(goma_ctx->fem.esp->rho, goma_ctx->fem.bf[v]->grad_phi, dofs,
                        goma_ctx->fem.fv->grad_rho);
#endif
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][DENSITY_EQN] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_rho[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[TFMP_PRES]) {
    v = TFMP_PRES;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
#ifdef DO_NO_UNROLL
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_tfmp_pres[p] = 0.0;

      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_tfmp_pres[p] +=
            *goma_ctx->fem.esp->tfmp_pres[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        goma_ctx->fem.fv_old->grad_tfmp_pres[p] +=
            *goma_ctx->fem.esp_old->tfmp_pres[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
#else
    grad_scalar_fv_fill(goma_ctx->fem.esp->tfmp_pres, goma_ctx->fem.bf[v]->grad_phi, dofs,
                        goma_ctx->fem.fv->grad_tfmp_pres);
#endif
  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][TFMP_PRES] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_tfmp_pres[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[TFMP_SAT]) {
    v = TFMP_SAT;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
#ifdef DO_NO_UNROLL
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_tfmp_sat[p] = 0.0;

      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_tfmp_sat[p] +=
            *goma_ctx->fem.esp->tfmp_sat[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        goma_ctx->fem.fv_old->grad_tfmp_sat[p] +=
            *goma_ctx->fem.esp_old->tfmp_sat[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
#else
    grad_scalar_fv_fill(goma_ctx->fem.esp->tfmp_sat, goma_ctx->fem.bf[v]->grad_phi, dofs,
                        goma_ctx->fem.fv->grad_tfmp_sat);
#endif
  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][TFMP_SAT] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_tfmp_sat[p] = 0.0;
  }

  if (goma_ctx->fem.pd->gv[RESTIME]) {
    v = RESTIME;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
#ifdef DO_NO_UNROLL
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_restime[p] = 0.0;
      goma_ctx->fem.fv_old->grad_restime[p] = 0.0;
      for (i = 0; i < dofs; i++) {
        goma_ctx->fem.fv->grad_restime[p] +=
            *goma_ctx->fem.esp->restime[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
        goma_ctx->fem.fv_old->grad_restime[p] +=
            *goma_ctx->fem.esp_old->restime[i] * goma_ctx->fem.bf[v]->grad_phi[i][p];
      }
    }
#else
    grad_scalar_fv_fill(goma_ctx->fem.esp->restime, goma_ctx->fem.bf[v]->grad_phi, dofs,
                        goma_ctx->fem.fv->grad_restime);
#endif
  } else if (zero_unused_grads && goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][RESTIME] == -1) {
    for (p = 0; p < VIM; p++)
      goma_ctx->fem.fv->grad_restime[p] = 0.0;
  }

  /*
   * EM Wave Vector Gradients
   */
  if (goma_ctx->fem.pd->gv[EM_E1_REAL] || goma_ctx->fem.pd->gv[EM_E2_REAL] ||
      goma_ctx->fem.pd->gv[EM_E3_REAL]) {
    v = EM_E1_REAL;
    if (goma_ctx->fem.bf[v]->interpolation != I_N1) {
      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

      // grad_vector_fv_fill(esp->em_er, bf[v]->grad_phi_e, dofs, fv->grad_em_er);
      for (p = 0; p < dim; p++) {
        for (q = 0; q < dim; q++) {
          goma_ctx->fem.fv->grad_em_er[p][q] = 0.0;
          goma_ctx->fem.fv_old->grad_em_er[p][q] = 0.0;
          for (r = 0; r < dim; r++) {
            for (i = 0; i < dofs; i++) {
              goma_ctx->fem.fv->grad_em_er[p][q] +=
                  (*goma_ctx->fem.esp->em_er[r][i]) * goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
              if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
                goma_ctx->fem.fv_old->grad_em_er[p][q] +=
                    (*goma_ctx->fem.esp_old->em_er[r][i]) *
                    goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
              }
            }
          }
        }
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_E1_REAL] == -1 &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_E2_REAL] == -1 &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_E3_REAL] == -1) {
    for (p = 0; p < DIM; p++) {
      for (q = 0; q < DIM; q++) {
        goma_ctx->fem.fv->grad_em_er[p][q] = 0.0;
      }
    }
  }

  if (goma_ctx->fem.pd->gv[EM_E1_IMAG] || goma_ctx->fem.pd->gv[EM_E2_IMAG] ||
      goma_ctx->fem.pd->gv[EM_E3_IMAG]) {
    v = EM_E1_IMAG;
    if (goma_ctx->fem.bf[v]->interpolation != I_N1) {
      dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

      // grad_vector_fv_fill(esp->em_ei, bf[v]->grad_phi_e, dofs, fv->grad_em_ei);
      for (p = 0; p < dim; p++) {
        for (q = 0; q < dim; q++) {
          goma_ctx->fem.fv->grad_em_ei[p][q] = 0.0;
          goma_ctx->fem.fv_old->grad_em_ei[p][q] = 0.0;
          for (r = 0; r < dim; r++) {
            for (i = 0; i < dofs; i++) {
              goma_ctx->fem.fv->grad_em_ei[p][q] +=
                  (*goma_ctx->fem.esp->em_ei[r][i]) * goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
              if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
                goma_ctx->fem.fv_old->grad_em_ei[p][q] +=
                    (*goma_ctx->fem.esp_old->em_ei[r][i]) *
                    goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
              }
            }
          }
        }
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_E1_IMAG] == -1 &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_E2_IMAG] == -1 &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_E3_IMAG] == -1) {
    for (p = 0; p < DIM; p++) {
      for (q = 0; q < DIM; q++) {
        goma_ctx->fem.fv->grad_em_ei[p][q] = 0.0;
      }
    }
  }
  if (goma_ctx->fem.pd->gv[EM_H1_REAL] || goma_ctx->fem.pd->gv[EM_H2_REAL] ||
      goma_ctx->fem.pd->gv[EM_H3_REAL]) {
    v = EM_H1_REAL;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

    // grad_vector_fv_fill(esp->em_hr, bf[v]->grad_phi_e, dofs, fv->grad_em_hr);
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        goma_ctx->fem.fv->grad_em_hr[p][q] = 0.0;
        goma_ctx->fem.fv_old->grad_em_hr[p][q] = 0.0;
        for (r = 0; r < WIM; r++) {
          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->grad_em_hr[p][q] +=
                (*goma_ctx->fem.esp->em_hr[r][i]) * goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
            if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
              goma_ctx->fem.fv_old->grad_em_hr[p][q] += (*goma_ctx->fem.esp_old->em_hr[r][i]) *
                                                        goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
            }
          }
        }
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_H1_REAL] == -1 &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_H2_REAL] == -1 &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_H3_REAL] == -1) {
    for (p = 0; p < DIM; p++) {
      for (q = 0; q < DIM; q++) {
        goma_ctx->fem.fv->grad_em_hr[p][q] = 0.0;
      }
    }
  }
  if (goma_ctx->fem.pd->gv[EM_H1_IMAG] || goma_ctx->fem.pd->gv[EM_H2_IMAG] ||
      goma_ctx->fem.pd->gv[EM_H3_IMAG]) {
    v = EM_H1_IMAG;
    dofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

    // grad_vector_fv_fill(esp->em_hi, bf[v]->grad_phi_e, dofs, fv->grad_em_hi);
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        goma_ctx->fem.fv->grad_em_hi[p][q] = 0.0;
        goma_ctx->fem.fv_old->grad_em_hi[p][q] = 0.0;
        for (r = 0; r < WIM; r++) {
          for (i = 0; i < dofs; i++) {
            goma_ctx->fem.fv->grad_em_hi[p][q] +=
                (*goma_ctx->fem.esp->em_hi[r][i]) * goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
            if (goma_ctx->fem.pd->TimeIntegration != STEADY) {
              goma_ctx->fem.fv_old->grad_em_hi[p][q] += (*goma_ctx->fem.esp_old->em_hi[r][i]) *
                                                        goma_ctx->fem.bf[v]->grad_phi_e[i][r][p][q];
            }
          }
        }
      }
    }
  } else if (zero_unused_grads &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_H1_IMAG] == -1 &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_H2_IMAG] == -1 &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_H3_IMAG] == -1) {
    for (p = 0; p < DIM; p++) {
      for (q = 0; q < DIM; q++) {
        goma_ctx->fem.fv->grad_em_hi[p][q] = 0.0;
      }
    }
  }

  /* grad(wall_distance)
   *
   */
  if (goma_ctx->fem.upd->turbulent_info->use_internal_wall_distance) {
    if (goma_ctx->fem.pd->gv[goma_ctx->fem.pd->ShapeVar]) {
      dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[goma_ctx->fem.pd->ShapeVar];
      for (p = 0; p < VIM; p++) {
        goma_ctx->fem.fv->grad_wall_distance[p] = 0.0;
        for (i = 0; i < dofs; i++) {
          goma_ctx->fem.fv->grad_wall_distance[p] +=
              goma_ctx->fem.upd->turbulent_info
                  ->wall_distances[goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]
                                       ->gnn_list[goma_ctx->fem.pd->ShapeVar][i]] *
              goma_ctx->fem.bf[goma_ctx->fem.pd->ShapeVar]->grad_phi[i][p];
        }
      }
    }
  } else if (zero_unused_grads && !goma_ctx->fem.upd->turbulent_info->use_internal_wall_distance) {
    for (p = 0; p < VIM; p++) {
      goma_ctx->fem.fv->grad_wall_distance[p] = 0.0;
    }
  }

  /*
   * External
   */
  if (goma_ctx->fem.efv->ev) {
    v = EXTERNAL;
    for (w = 0; w < goma_ctx->fem.efv->Num_external_field; w++) {
      dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof_ext[w];
      if (goma_ctx->fem.efv->i[w] != I_TABLE) {
        if (strcmp(goma_ctx->fem.efv->name[w], "F1") == 0 && !goma_ctx->fem.pd->gv[PHASE1]) {

          /* load up the gradient of the the phase function variables
           * Currently, this is the only field whose gradient is computed when
           * is applied as externally
           */
          grad_scalar_fv_fill(goma_ctx->fem.evp->external_field[w], goma_ctx->fem.bfex[v]->grad_phi,
                              dofs, goma_ctx->fem.fv->grad_pF[0]);
        } else {
          for (p = 0; p < VIM; p++) {
            goma_ctx->fem.fv->grad_ext_field[w][p] = 0.0;
            for (i = 0; i < dofs; i++) {
              goma_ctx->fem.fv->grad_ext_field[w][p] +=
                  *goma_ctx->fem.evp->external_field[w][i] * goma_ctx->fem.bfex[w]->grad_phi[i][p];
            }
          }
        }
      }
    }
  }

  zero_unused_grads = FALSE;
  return (status);
}

/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/

int load_fv_mesh_derivs(int okToZero)

/*******************************************************************************
 *
 * load_fv_mesh_derivs() -- ld mesh derivatives of field var grads at gauss pt
 *
 * input: (assume the appropriate parts of esp, bf, ei, pd are filled in)
 * ------
 *
 *
 * output: ( this routine fills in the following parts of the fv structure)
 * -------
 *	fv->
 *  		d_grad_T_dmesh[p] [b][j] 	d( e_p . grad(T) )
 * 						------------------
 * 						    d ( d_b,j )
 *
 *		d_grad_P_dmesh[p] [b][j] 	d( e_p . grad(P) )
 * 						------------------
 * 						    d ( d_b,j )
 *
 *		d_grad_c_dmesh[p][w] [b][j] 	d( e_p . grad(c_w) )
 * 						------------------
 * 						    d ( d_b,j )
 *
 *		d_grad_p_liq_dmesh[p] [b][j] 	d( e_p . grad(p_liq) )
 * 						------------------
 * 						    d ( d_b,j )
 *
 *		d_div_v_dmesh [b][j] 		d( grad.v )
 * 						-----------
 * 						d ( d_b,j )
 *
 *		d_grad_v_dmesh[p][q] [b][j] 	d( e_p e_q : grad(v) )
 * 						------------------
 * 						    d ( d_b,j )
 *
 *		d_grad_X_dmesh[p][q] [b][j] 	d( e_p e_q : grad(X) )
 * 						------------------
 * 						    d ( d_b,j )
 *
 *		d_grad_pv_dmesh[p][q] [b][j] 	d( e_p e_q : grad(pv) )
 * 						------------------
 * 						    d ( d_b,j )
 *
 *		d_div_d_dmesh [b][j] 		d( grad.d )
 * 						-----------
 * 						d ( d_b,j )
 *
 *		d_grad_d_dmesh[p][q] [b][j] 	d( e_p e_q : grad(d) )
 * 						------------------
 * 						    d ( d_b,j )
 *
 *		d_grad_S_dmesh[p][q][r] [b][j] 	d( e_p e_q e_r : grad(S) )
 * 						------------------
 * 						    d ( d_b,j )
 *
 *		d_grad_G_dmesh[p][q][r] [b][j] 	d( e_p e_q e_r : grad(G) )
 * 						------------------
 * 						    d ( d_b,j )
 *
 *		d_grad_Gt_dmesh[p][q][r] [b][j] d( e_p e_q e_r : grad(G)^t )
 * 						------------------
 * 						    d ( d_b,j )
 *
 *		d_grad_vd_dmesh[p][q] [b][j] 	d( e_p e_q : grad(vd) )
 * 						------------------
 * 						    d ( d_b,j )
 *
 * NOTE: To accommodate shell elements, this function has been modified
 *       so that fv variables are not zeroed out when they are active
 *       on an element block other than the current one.
 *       The check done for variable v is then:
 *          if ( pd->v[pg->imtrx][v] || upd->vp[pg->imtrx][v] == -1 )
 *       In many cases below, this conditional zeroing is done in
 *       a separate small loop before the main one.
 *
 *
 * Return values:
 *		0 -- if things went OK
 *	       -1 -- if a problem occurred
 *
 *
 * Notes:
 * ------
 *		[1] There is an extra Langrangian contribution for the mesh
 *		    displacment unknowns, in addition to the parts that the
 *		    have to do with spatial gradient operators over a
 *		    deforming mesh that all variables have.
 *
 *
 * Created:	Fri Mar 18 10:18:03 MST 1994 pasacki@sandia.gov
 *
 * Modified:	Tue Feb 21 11:16 MST 1995 pasacki@sandia.gov
 ****************************************************************************/
{
  int v;    /* variable type indicator */
  int i, j; /* index */
  int b;
  int p, q, r, s; /* dimension index */
  int vdofs = 0;  /* degrees of freedom for var  in the elem */
  int mdofs;      /* degrees of freedom for mesh in the elem */
  int w;          /* concentration species */
  int status;
  int dimNonSym; /* # of dimensions that don't have symmetry */
  int dim;       /* # dimensions in the physical mesh */
  unsigned int siz;
  int mode; /* modal counter */
  int transient_run, discontinuous_porous_media, ve_solid;

  dbl T_i, v_ri, P_i, d_ri, F_i, d_dot_ri;
  dbl em_er_ri, em_ei_ri, em_hr_ri, em_hi_ri;

  struct Basis_Functions *bfm, *bfv; /* For mesh variables. */

  static int is_initialized = FALSE;
  int VIMis3;

  status = 0;

  VIMis3 = (VIM == 3) ? TRUE : FALSE;

  /*
   * If this is not a deforming mesh problem, then
   * leave now.
   */
  int imtrx;
  int mesh_on = FALSE;
  for (imtrx = 0; imtrx < goma_ctx->fem.upd->Total_Num_Matrices; imtrx++) {
    if (goma_ctx->fem.ei[imtrx]->deforming_mesh) {
      mesh_on = TRUE;
    }
  }
  if (!mesh_on) {
    return status;
  }

  dim = goma_ctx->fem.pd->Num_Dim;
  dimNonSym = goma_ctx->fem.pd->Num_Dim;

  mdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[R_MESH1]]->dof[R_MESH1];

  bfm = goma_ctx->fem.bf[R_MESH1];

  transient_run = goma_ctx->fem.pd->TimeIntegration != STEADY;
  discontinuous_porous_media = goma_ctx->fem.mp->PorousMediaType != CONTINUOUS;
  ve_solid = goma_ctx->fem.cr->MeshFluxModel == KELVIN_VOIGT;

  okToZero = FALSE;

  /*
   * d(grad(T))/dmesh
   */

  if (goma_ctx->fem.pd->gv[TEMPERATURE]) {
    v = TEMPERATURE;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_T_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->T[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_T_dmesh[p][b][j] += T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->T[0];

      goma_ctx->fem.fv->d_grad_T_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_T_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_T_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_T_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_T_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_T_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_T_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_T_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_T_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->T[i];

        goma_ctx->fem.fv->d_grad_T_dmesh[0][0][j] += T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_T_dmesh[1][1][j] += T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_T_dmesh[1][0][j] += T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_T_dmesh[0][1][j] += T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_T_dmesh[2][2][j] += T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_T_dmesh[2][0][j] += T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_T_dmesh[2][1][j] += T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_T_dmesh[0][2][j] += T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_T_dmesh[1][2][j] += T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][TEMPERATURE] != -1 && !is_initialized &&
             okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_T_dmesh[0][0][0]), 0, siz);
  }

  int mom;
  for (mom = 0; mom < MAX_MOMENTS; mom++) {
    v = MOMENT0 + mom;
    if (goma_ctx->fem.pd->gv[v]) {
      bfv = goma_ctx->fem.bf[v];
      vdofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];

      siz = sizeof(double) * DIM * DIM * MDE;
      memset(&(goma_ctx->fem.fv->d_grad_moment_dmesh[mom][0][0][0]), 0, siz);
      for (i = 0; i < vdofs; i++) {
        for (p = 0; p < dimNonSym; p++) {
          for (b = 0; b < dim; b++) {
            for (j = 0; j < mdofs; j++) {
              goma_ctx->fem.fv->d_grad_moment_dmesh[mom][p][b][j] +=
                  (*goma_ctx->fem.esp->moment[mom][i]) * bfv->d_grad_phi_dmesh[i][p][b][j];
            }
          }
        }
      }

    } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][v] != -1 && !is_initialized &&
               okToZero) {
      siz = sizeof(double) * DIM * DIM * MDE;
      memset(&(goma_ctx->fem.fv->d_grad_moment_dmesh[mom][0][0][0]), 0, siz);
    }
  }

  /*
   * d(grad(V))/dmesh
   */
  if (goma_ctx->fem.pd->gv[VOLTAGE]) {
    v = VOLTAGE;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_V_dmesh[0][0][0]), 0, siz);
    for (p = 0; p < dimNonSym; p++) {
      for (b = 0; b < dim; b++) {
        for (j = 0; j < mdofs; j++) {
          for (i = 0; i < vdofs; i++) {
            goma_ctx->fem.fv->d_grad_V_dmesh[p][b][j] +=
                *goma_ctx->fem.esp->V[i] * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      goma_ctx->fem.fv->d_grad_V_dmesh[0][0][j] =
          *goma_ctx->fem.esp->V[0] * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_V_dmesh[1][1][j] =
          *goma_ctx->fem.esp->V[0] * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_V_dmesh[1][0][j] =
          *goma_ctx->fem.esp->V[0] * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_V_dmesh[0][1][j] =
          *goma_ctx->fem.esp->V[0] * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_V_dmesh[2][2][j] =
            *goma_ctx->fem.esp->V[0] * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_V_dmesh[2][0][j] =
            *goma_ctx->fem.esp->V[0] * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_V_dmesh[2][1][j] =
            *goma_ctx->fem.esp->V[0] * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_V_dmesh[0][2][j] =
            *goma_ctx->fem.esp->V[0] * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_V_dmesh[1][2][j] =
            *goma_ctx->fem.esp->V[0] * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        goma_ctx->fem.fv->d_grad_V_dmesh[0][0][j] +=
            *goma_ctx->fem.esp->V[i] * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_V_dmesh[1][1][j] +=
            *goma_ctx->fem.esp->V[i] * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_V_dmesh[1][0][j] +=
            *goma_ctx->fem.esp->V[i] * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_V_dmesh[0][1][j] +=
            *goma_ctx->fem.esp->V[i] * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_V_dmesh[2][2][j] +=
              *goma_ctx->fem.esp->V[i] * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_V_dmesh[2][0][j] +=
              *goma_ctx->fem.esp->V[i] * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_V_dmesh[2][1][j] +=
              *goma_ctx->fem.esp->V[i] * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_V_dmesh[0][2][j] +=
              *goma_ctx->fem.esp->V[i] * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_V_dmesh[1][2][j] +=
              *goma_ctx->fem.esp->V[i] * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][VOLTAGE] != -1 && !is_initialized &&
             okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_V_dmesh[0][0][0]), 0, siz);
  }

  /*
   * d(grad(sh_K))/dmesh
   */

  if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][SHELL_CURVATURE]) {
    v = SHELL_CURVATURE;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_K_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->sh_K[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_sh_K_dmesh[p][b][j] += T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->sh_K[0];

      goma_ctx->fem.fv->d_grad_sh_K_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_sh_K_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_sh_K_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_sh_K_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_sh_K_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_sh_K_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_sh_K_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_sh_K_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_sh_K_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->sh_K[i];

        goma_ctx->fem.fv->d_grad_sh_K_dmesh[0][0][j] += T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_sh_K_dmesh[1][1][j] += T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_sh_K_dmesh[1][0][j] += T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_sh_K_dmesh[0][1][j] += T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_sh_K_dmesh[2][2][j] += T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_sh_K_dmesh[2][0][j] += T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_sh_K_dmesh[2][1][j] += T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_sh_K_dmesh[0][2][j] += T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_sh_K_dmesh[1][2][j] += T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_CURVATURE] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_K_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][SHELL_CURVATURE2]) {
    v = SHELL_CURVATURE2;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_K2_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->sh_K2[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_sh_K2_dmesh[p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->sh_K2[0];

      goma_ctx->fem.fv->d_grad_sh_K2_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_sh_K2_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_sh_K2_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_sh_K2_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_sh_K2_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_sh_K2_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_sh_K2_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_sh_K2_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_sh_K2_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->sh_K2[i];

        goma_ctx->fem.fv->d_grad_sh_K2_dmesh[0][0][j] += T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_sh_K2_dmesh[1][1][j] += T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_sh_K2_dmesh[1][0][j] += T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_sh_K2_dmesh[0][1][j] += T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_sh_K2_dmesh[2][2][j] += T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_sh_K2_dmesh[2][0][j] += T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_sh_K2_dmesh[2][1][j] += T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_sh_K2_dmesh[0][2][j] += T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_sh_K2_dmesh[1][2][j] += T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_CURVATURE2] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_K2_dmesh[0][0][0]), 0, siz);
  }

  /*
   * d(grad(apr))/dmesh
   */
  if (goma_ctx->fem.pd->gv[ACOUS_PREAL]) {
    v = ACOUS_PREAL;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_apr_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->apr[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_apr_dmesh[p][b][j] += T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->apr[0];

      goma_ctx->fem.fv->d_grad_apr_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_apr_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_apr_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_apr_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_apr_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_apr_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_apr_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_apr_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_apr_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->apr[i];

        goma_ctx->fem.fv->d_grad_apr_dmesh[0][0][j] += T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_apr_dmesh[1][1][j] += T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_apr_dmesh[1][0][j] += T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_apr_dmesh[0][1][j] += T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_apr_dmesh[2][2][j] += T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_apr_dmesh[2][0][j] += T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_apr_dmesh[2][1][j] += T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_apr_dmesh[0][2][j] += T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_apr_dmesh[1][2][j] += T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][ACOUS_PREAL] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_apr_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[ACOUS_PIMAG]) {
    v = ACOUS_PIMAG;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_api_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->api[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_api_dmesh[p][b][j] += T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->api[0];

      goma_ctx->fem.fv->d_grad_api_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_api_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_api_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_api_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_api_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_api_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_api_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_api_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_api_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->api[i];

        goma_ctx->fem.fv->d_grad_api_dmesh[0][0][j] += T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_api_dmesh[1][1][j] += T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_api_dmesh[1][0][j] += T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_api_dmesh[0][1][j] += T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_api_dmesh[2][2][j] += T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_api_dmesh[2][0][j] += T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_api_dmesh[2][1][j] += T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_api_dmesh[0][2][j] += T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_api_dmesh[1][2][j] += T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][ACOUS_PIMAG] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_api_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[ACOUS_REYN_STRESS]) {
    v = ACOUS_REYN_STRESS;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_ars_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->ars[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_ars_dmesh[p][b][j] += T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->ars[0];

      goma_ctx->fem.fv->d_grad_ars_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_ars_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_ars_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_ars_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_ars_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_ars_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_ars_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_ars_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_ars_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->ars[i];

        goma_ctx->fem.fv->d_grad_ars_dmesh[0][0][j] += T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_ars_dmesh[1][1][j] += T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_ars_dmesh[1][0][j] += T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_ars_dmesh[0][1][j] += T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_ars_dmesh[2][2][j] += T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_ars_dmesh[2][0][j] += T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_ars_dmesh[2][1][j] += T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_ars_dmesh[0][2][j] += T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_ars_dmesh[1][2][j] += T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][ACOUS_REYN_STRESS] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_ars_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[SHELL_BDYVELO]) {
    v = SHELL_BDYVELO;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_bv_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->sh_bv[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_sh_bv_dmesh[p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->sh_bv[0];

      goma_ctx->fem.fv->d_grad_sh_bv_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_sh_bv_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_sh_bv_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_sh_bv_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_sh_bv_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_sh_bv_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_sh_bv_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_sh_bv_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_sh_bv_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->sh_bv[i];

        goma_ctx->fem.fv->d_grad_sh_bv_dmesh[0][0][j] += T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_sh_bv_dmesh[1][1][j] += T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_sh_bv_dmesh[1][0][j] += T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_sh_bv_dmesh[0][1][j] += T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_sh_bv_dmesh[2][2][j] += T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_sh_bv_dmesh[2][0][j] += T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_sh_bv_dmesh[2][1][j] += T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_sh_bv_dmesh[0][2][j] += T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_sh_bv_dmesh[1][2][j] += T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_BDYVELO] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_bv_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[LUBP]) {
    v = LUBP;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_lubp_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->lubp[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_lubp_dmesh[p][b][j] += T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->lubp[0];

      goma_ctx->fem.fv->d_grad_lubp_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_lubp_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_lubp_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_lubp_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_lubp_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_lubp_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_lubp_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_lubp_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_lubp_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->lubp[i];

        goma_ctx->fem.fv->d_grad_lubp_dmesh[0][0][j] += T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_lubp_dmesh[1][1][j] += T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_lubp_dmesh[1][0][j] += T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_lubp_dmesh[0][1][j] += T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_lubp_dmesh[2][2][j] += T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_lubp_dmesh[2][0][j] += T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_lubp_dmesh[2][1][j] += T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_lubp_dmesh[0][2][j] += T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_lubp_dmesh[1][2][j] += T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][LUBP] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_lubp_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[LUBP_2]) {
    v = LUBP_2;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_lubp_2_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->lubp_2[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_lubp_2_dmesh[p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->lubp_2[0];

      goma_ctx->fem.fv->d_grad_lubp_2_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_lubp_2_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_lubp_2_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_lubp_2_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_lubp_2_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_lubp_2_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_lubp_2_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_lubp_2_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_lubp_2_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->lubp_2[i];

        goma_ctx->fem.fv->d_grad_lubp_2_dmesh[0][0][j] += T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_lubp_2_dmesh[1][1][j] += T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_lubp_2_dmesh[1][0][j] += T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_lubp_2_dmesh[0][1][j] += T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_lubp_2_dmesh[2][2][j] += T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_lubp_2_dmesh[2][0][j] += T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_lubp_2_dmesh[2][1][j] += T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_lubp_2_dmesh[0][2][j] += T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_lubp_2_dmesh[1][2][j] += T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][LUBP_2] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_lubp_2_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[SHELL_PRESS_OPEN]) {
    v = SHELL_PRESS_OPEN;
    bfv = goma_ctx->fem.bf[v];
#ifdef DO_NOT_UNROLL
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->sh_p_open[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->sh_p_open[0];

      goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->sh_p_open[i];

        goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[0][0][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[1][1][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[1][0][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[0][1][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[2][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[2][0][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[2][1][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[0][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[1][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_PRESS_OPEN] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_p_open_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[SHELL_PRESS_OPEN_2]) {
    v = SHELL_PRESS_OPEN_2;
    bfv = goma_ctx->fem.bf[v];
#ifdef DO_NOT_UNROLL
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->sh_p_open_2[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->sh_p_open_2[0];

      goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[2][2][j] =
            T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[2][0][j] =
            T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[2][1][j] =
            T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[0][2][j] =
            T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[1][2][j] =
            T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->sh_p_open_2[i];

        goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[0][0][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[1][1][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[1][0][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[0][1][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[2][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[2][0][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[2][1][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[0][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[1][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_PRESS_OPEN_2] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_p_open_2_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[SHELL_FILMP]) {
    v = SHELL_FILMP;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_fp_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->sh_fp[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_sh_fp_dmesh[p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->sh_fp[0];

      goma_ctx->fem.fv->d_grad_sh_fp_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_sh_fp_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_sh_fp_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_sh_fp_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_sh_fp_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_sh_fp_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_sh_fp_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_sh_fp_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_sh_fp_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->sh_fp[i];

        goma_ctx->fem.fv->d_grad_sh_fp_dmesh[0][0][j] += T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_sh_fp_dmesh[1][1][j] += T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_sh_fp_dmesh[1][0][j] += T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_sh_fp_dmesh[0][1][j] += T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_sh_fp_dmesh[2][2][j] += T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_sh_fp_dmesh[2][0][j] += T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_sh_fp_dmesh[2][1][j] += T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_sh_fp_dmesh[0][2][j] += T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_sh_fp_dmesh[1][2][j] += T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_FILMP] != -1) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_fp_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[SHELL_FILMH]) {
    v = SHELL_FILMH;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_fh_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->sh_fh[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_sh_fh_dmesh[p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->sh_fh[0];

      goma_ctx->fem.fv->d_grad_sh_fh_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_sh_fh_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_sh_fh_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_sh_fh_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_sh_fh_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_sh_fh_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_sh_fh_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_sh_fh_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_sh_fh_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->sh_fh[i];

        goma_ctx->fem.fv->d_grad_sh_fh_dmesh[0][0][j] += T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_sh_fh_dmesh[1][1][j] += T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_sh_fh_dmesh[1][0][j] += T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_sh_fh_dmesh[0][1][j] += T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_sh_fh_dmesh[2][2][j] += T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_sh_fh_dmesh[2][0][j] += T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_sh_fh_dmesh[2][1][j] += T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_sh_fh_dmesh[0][2][j] += T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_sh_fh_dmesh[1][2][j] += T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_FILMH] != -1) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_fh_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[SHELL_PARTC]) {
    v = SHELL_PARTC;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_pc_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->sh_pc[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_sh_pc_dmesh[p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->sh_pc[0];

      goma_ctx->fem.fv->d_grad_sh_pc_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_sh_pc_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_sh_pc_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_sh_pc_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_sh_pc_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_sh_pc_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_sh_pc_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_sh_pc_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_sh_pc_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->sh_pc[i];

        goma_ctx->fem.fv->d_grad_sh_pc_dmesh[0][0][j] += T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_sh_pc_dmesh[1][1][j] += T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_sh_pc_dmesh[1][0][j] += T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_sh_pc_dmesh[0][1][j] += T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_sh_pc_dmesh[2][2][j] += T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_sh_pc_dmesh[2][0][j] += T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_sh_pc_dmesh[2][1][j] += T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_sh_pc_dmesh[0][2][j] += T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_sh_pc_dmesh[1][2][j] += T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_PARTC] != -1) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_pc_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[SHELL_TEMPERATURE]) {
    v = SHELL_TEMPERATURE;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_t_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->sh_t[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_sh_t_dmesh[p][b][j] += T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->sh_t[0];

      goma_ctx->fem.fv->d_grad_sh_t_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_sh_t_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_sh_t_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_sh_t_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_sh_t_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_sh_t_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_sh_t_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_sh_t_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_sh_t_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->sh_t[i];

        goma_ctx->fem.fv->d_grad_sh_t_dmesh[0][0][j] += T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_sh_t_dmesh[1][1][j] += T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_sh_t_dmesh[1][0][j] += T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_sh_t_dmesh[0][1][j] += T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_sh_t_dmesh[2][2][j] += T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_sh_t_dmesh[2][0][j] += T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_sh_t_dmesh[2][1][j] += T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_sh_t_dmesh[0][2][j] += T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_sh_t_dmesh[1][2][j] += T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_TEMPERATURE] != -1) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_t_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[EDDY_NU]) {
    v = EDDY_NU;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->eddy_nu[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->eddy_nu[0];

      goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->eddy_nu[i];

        goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[0][0][j] += T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[1][1][j] += T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[1][0][j] += T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[0][1][j] += T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[2][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[2][0][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[2][1][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[0][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[1][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EDDY_NU] != -1) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_eddy_nu_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[TURB_K]) {
    v = TURB_K;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_turb_k_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->turb_k[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_turb_k_dmesh[p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->turb_k[0];

      goma_ctx->fem.fv->d_grad_turb_k_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_turb_k_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_turb_k_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_turb_k_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_turb_k_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_turb_k_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_turb_k_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_turb_k_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_turb_k_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->turb_k[i];

        goma_ctx->fem.fv->d_grad_turb_k_dmesh[0][0][j] += T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_turb_k_dmesh[1][1][j] += T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_turb_k_dmesh[1][0][j] += T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_turb_k_dmesh[0][1][j] += T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_turb_k_dmesh[2][2][j] += T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_turb_k_dmesh[2][0][j] += T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_turb_k_dmesh[2][1][j] += T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_turb_k_dmesh[0][2][j] += T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_turb_k_dmesh[1][2][j] += T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][TURB_K] != -1) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_turb_k_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[TURB_OMEGA]) {
    v = TURB_OMEGA;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_turb_omega_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->turb_omega[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_turb_omega_dmesh[p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->turb_omega[0];

      goma_ctx->fem.fv->d_grad_turb_omega_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_turb_omega_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_turb_omega_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_turb_omega_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_turb_omega_dmesh[2][2][j] =
            T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_turb_omega_dmesh[2][0][j] =
            T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_turb_omega_dmesh[2][1][j] =
            T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_turb_omega_dmesh[0][2][j] =
            T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_turb_omega_dmesh[1][2][j] =
            T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->turb_omega[i];

        goma_ctx->fem.fv->d_grad_turb_omega_dmesh[0][0][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_turb_omega_dmesh[1][1][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_turb_omega_dmesh[1][0][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_turb_omega_dmesh[0][1][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_turb_omega_dmesh[2][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_turb_omega_dmesh[2][0][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_turb_omega_dmesh[2][1][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_turb_omega_dmesh[0][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_turb_omega_dmesh[1][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][TURB_OMEGA] != -1) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_turb_omega_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[LIGHT_INTP]) {
    v = LIGHT_INTP;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_poynt_dmesh[0][0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->poynt[0][i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_poynt_dmesh[0][p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][LIGHT_INTP] != -1) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_poynt_dmesh[0][0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[LIGHT_INTM]) {
    v = LIGHT_INTM;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_poynt_dmesh[1][0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->poynt[1][i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_poynt_dmesh[1][p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][LIGHT_INTM] != -1) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_poynt_dmesh[1][0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[LIGHT_INTD]) {
    v = LIGHT_INTD;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_poynt_dmesh[2][0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->poynt[2][i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_poynt_dmesh[2][p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][LIGHT_INTD] != -1) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_poynt_dmesh[2][0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[TFMP_PRES]) {
    v = TFMP_PRES;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_tfmp_pres_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->tfmp_pres[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_tfmp_pres_dmesh[p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][TFMP_PRES] != -1) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_tfmp_pres_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[TFMP_SAT]) {
    v = TFMP_SAT;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_tfmp_sat_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->tfmp_sat[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_tfmp_sat_dmesh[p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][TFMP_PRES] != -1) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_tfmp_pres_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[RESTIME]) {
    v = RESTIME;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_restime_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->restime[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_restime_dmesh[p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][RESTIME] != -1) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_restime_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[SHELL_LUB_CURV]) {
    v = SHELL_LUB_CURV;
    bfv = goma_ctx->fem.bf[v];
#ifdef DO_NOT_UNROLL
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->sh_l_curv[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->sh_l_curv[0];

      goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[2][2][j] = T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[2][0][j] = T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[2][1][j] = T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[0][2][j] = T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[1][2][j] = T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->sh_l_curv[i];

        goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[0][0][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[1][1][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[1][0][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[0][1][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[2][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[2][0][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[2][1][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[0][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[1][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_LUB_CURV] != -1) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_l_curv_dmesh[0][0][0]), 0, siz);
  }

  if (goma_ctx->fem.pd->gv[SHELL_LUB_CURV_2]) {
    v = SHELL_LUB_CURV_2;
    bfv = goma_ctx->fem.bf[v];
#ifdef DO_NOT_UNROLL
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[0][0][0]), 0, siz);
    for (i = 0; i < vdofs; i++) {
      T_i = *goma_ctx->fem.esp->sh_l_curv_2[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[p][b][j] +=
                T_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      T_i = *goma_ctx->fem.esp->sh_l_curv_2[0];

      goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[0][0][j] = T_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[1][1][j] = T_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[1][0][j] = T_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[0][1][j] = T_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[2][2][j] =
            T_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[2][0][j] =
            T_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[2][1][j] =
            T_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[0][2][j] =
            T_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[1][2][j] =
            T_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        T_i = *goma_ctx->fem.esp->sh_l_curv[i];

        goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[0][0][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[1][1][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[1][0][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[0][1][j] +=
            T_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[2][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[2][0][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[2][1][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[0][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[1][2][j] +=
              T_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_LUB_CURV_2] != -1) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_l_curv_2_dmesh[0][0][0]), 0, siz);
  }

  /*
   * d(grad(qs))/dmesh
   */

  if (goma_ctx->fem.pd->gv[SURF_CHARGE]) {
    v = SURF_CHARGE;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_qs_dmesh[0][0][0]), 0, siz);
    for (p = 0; p < dimNonSym; p++) {
      for (b = 0; b < dim; b++) {
        for (j = 0; j < mdofs; j++) {
          for (i = 0; i < vdofs; i++) {
            goma_ctx->fem.fv->d_grad_qs_dmesh[p][b][j] +=
                *goma_ctx->fem.esp->qs[i] * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      goma_ctx->fem.fv->d_grad_qs_dmesh[0][0][j] =
          *goma_ctx->fem.esp->qs[0] * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_qs_dmesh[1][1][j] =
          *goma_ctx->fem.esp->qs[0] * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_qs_dmesh[1][0][j] =
          *goma_ctx->fem.esp->qs[0] * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_qs_dmesh[0][1][j] =
          *goma_ctx->fem.esp->qs[0] * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_qs_dmesh[2][2][j] =
            *goma_ctx->fem.esp->qs[0] * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_qs_dmesh[2][0][j] =
            *goma_ctx->fem.esp->qs[0] * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_qs_dmesh[2][1][j] =
            *goma_ctx->fem.esp->qs[0] * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_qs_dmesh[0][2][j] =
            *goma_ctx->fem.esp->qs[0] * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_qs_dmesh[1][2][j] =
            *goma_ctx->fem.esp->qs[0] * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        goma_ctx->fem.fv->d_grad_qs_dmesh[0][0][j] +=
            *goma_ctx->fem.esp->qs[i] * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_qs_dmesh[1][1][j] +=
            *goma_ctx->fem.esp->qs[i] * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_qs_dmesh[1][0][j] +=
            *goma_ctx->fem.esp->qs[i] * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_qs_dmesh[0][1][j] +=
            *goma_ctx->fem.esp->qs[i] * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_qs_dmesh[2][2][j] +=
              *goma_ctx->fem.esp->qs[i] * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_qs_dmesh[2][0][j] +=
              *goma_ctx->fem.esp->qs[i] * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_qs_dmesh[2][1][j] +=
              *goma_ctx->fem.esp->qs[i] * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_qs_dmesh[0][2][j] +=
              *goma_ctx->fem.esp->qs[i] * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_qs_dmesh[1][2][j] +=
              *goma_ctx->fem.esp->qs[i] * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SURF_CHARGE] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_qs_dmesh[0][0][0]), 0, siz);
  }

  /*
   * d(grad(sh_J))/dmesh
   */

  /* This is needed for bulk assembly of shell diffusion KBC */
  if (goma_ctx->fem.pd->gv[SHELL_DIFF_FLUX] &&
      goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[SHELL_DIFF_FLUX]]->dof[SHELL_DIFF_FLUX] >
          0) {
    v = SHELL_DIFF_FLUX;
    bfv = goma_ctx->fem.bf[v];
#ifdef DO_NOT_UNROLL
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_J_dmesh[0][0][0]), 0, siz);
    for (p = 0; p < dimNonSym; p++) {
      for (b = 0; b < dim; b++) {
        for (j = 0; j < mdofs; j++) {
          for (i = 0; i < vdofs; i++) {
            goma_ctx->fem.fv->d_grad_sh_J_dmesh[p][b][j] +=
                *goma_ctx->fem.esp->sh_J[i] * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      goma_ctx->fem.fv->d_grad_sh_J_dmesh[0][0][j] =
          *goma_ctx->fem.esp->sh_J[0] * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_sh_J_dmesh[1][1][j] =
          *goma_ctx->fem.esp->sh_J[0] * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_sh_J_dmesh[1][0][j] =
          *goma_ctx->fem.esp->sh_J[0] * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_sh_J_dmesh[0][1][j] =
          *goma_ctx->fem.esp->sh_J[0] * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_sh_J_dmesh[2][2][j] =
            *goma_ctx->fem.esp->sh_J[0] * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_sh_J_dmesh[2][0][j] =
            *goma_ctx->fem.esp->sh_J[0] * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_sh_J_dmesh[2][1][j] =
            *goma_ctx->fem.esp->sh_J[0] * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_sh_J_dmesh[0][2][j] =
            *goma_ctx->fem.esp->sh_J[0] * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_sh_J_dmesh[1][2][j] =
            *goma_ctx->fem.esp->sh_J[0] * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        goma_ctx->fem.fv->d_grad_sh_J_dmesh[0][0][j] +=
            *goma_ctx->fem.esp->sh_J[i] * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_sh_J_dmesh[1][1][j] +=
            *goma_ctx->fem.esp->sh_J[i] * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_sh_J_dmesh[1][0][j] +=
            *goma_ctx->fem.esp->sh_J[i] * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_sh_J_dmesh[0][1][j] +=
            *goma_ctx->fem.esp->sh_J[i] * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_sh_J_dmesh[2][2][j] +=
              *goma_ctx->fem.esp->sh_J[i] * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_sh_J_dmesh[2][0][j] +=
              *goma_ctx->fem.esp->sh_J[i] * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_sh_J_dmesh[2][1][j] +=
              *goma_ctx->fem.esp->sh_J[i] * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_sh_J_dmesh[0][2][j] +=
              *goma_ctx->fem.esp->sh_J[i] * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_sh_J_dmesh[1][2][j] +=
              *goma_ctx->fem.esp->sh_J[i] * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_DIFF_FLUX] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_J_dmesh[0][0][0]), 0, siz);
  }

  /*
   * d(grad(SH))/dmesh
   */
  if (goma_ctx->fem.pd->gv[SHEAR_RATE]) {
    v = SHEAR_RATE;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_SH_dmesh, 0, siz);

    for (p = 0; p < dimNonSym; p++) {
      for (b = 0; b < dim; b++) {
        for (j = 0; j < mdofs; j++) {
          for (i = 0; i < vdofs; i++) {
            goma_ctx->fem.fv->d_grad_SH_dmesh[p][b][j] +=
                *goma_ctx->fem.esp->SH[i] * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      goma_ctx->fem.fv->d_grad_SH_dmesh[0][0][j] =
          *goma_ctx->fem.esp->SH[0] * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_SH_dmesh[1][1][j] =
          *goma_ctx->fem.esp->SH[0] * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_SH_dmesh[1][0][j] =
          *goma_ctx->fem.esp->SH[0] * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_SH_dmesh[0][1][j] =
          *goma_ctx->fem.esp->SH[0] * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_SH_dmesh[2][2][j] =
            *goma_ctx->fem.esp->SH[0] * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_SH_dmesh[2][0][j] =
            *goma_ctx->fem.esp->SH[0] * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_SH_dmesh[2][1][j] =
            *goma_ctx->fem.esp->SH[0] * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_SH_dmesh[0][2][j] =
            *goma_ctx->fem.esp->SH[0] * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_SH_dmesh[1][2][j] =
            *goma_ctx->fem.esp->SH[0] * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        goma_ctx->fem.fv->d_grad_SH_dmesh[0][0][j] +=
            *goma_ctx->fem.esp->SH[i] * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_SH_dmesh[1][1][j] +=
            *goma_ctx->fem.esp->SH[i] * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_SH_dmesh[1][0][j] +=
            *goma_ctx->fem.esp->SH[i] * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_SH_dmesh[0][1][j] +=
            *goma_ctx->fem.esp->SH[i] * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_api_dmesh[2][2][j] +=
              *goma_ctx->fem.esp->SH[i] * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_api_dmesh[2][0][j] +=
              *goma_ctx->fem.esp->SH[i] * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_api_dmesh[2][1][j] +=
              *goma_ctx->fem.esp->SH[i] * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_api_dmesh[0][2][j] +=
              *goma_ctx->fem.esp->SH[i] * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_api_dmesh[1][2][j] +=
              *goma_ctx->fem.esp->SH[i] * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHEAR_RATE] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_SH_dmesh, 0, siz);
  }

  /*
   * d(grad(F))/dmesh
   */
  if (goma_ctx->fem.pd->gv[FILL]) {
    v = FILL;
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NO_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_F_dmesh, 0, siz);
    for (p = 0; p < dimNonSym; p++) {
      for (b = 0; b < dim; b++) {
        for (j = 0; j < mdofs; j++) {
          for (i = 0; i < vdofs; i++) {
            goma_ctx->fem.fv->d_grad_F_dmesh[p][b][j] +=
                *goma_ctx->fem.esp->F[i] * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      F_i = *goma_ctx->fem.esp->F[0];

      goma_ctx->fem.fv->d_grad_F_dmesh[0][0][j] =
          F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_F_dmesh[1][1][j] =
          F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_F_dmesh[1][0][j] =
          F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_F_dmesh[0][1][j] =
          F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_F_dmesh[2][2][j] =
            F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_F_dmesh[2][0][j] =
            F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_F_dmesh[2][1][j] =
            F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_F_dmesh[0][2][j] =
            F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_F_dmesh[1][2][j] =
            F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        F_i = *goma_ctx->fem.esp->F[i];

        goma_ctx->fem.fv->d_grad_F_dmesh[0][0][j] +=
            F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_F_dmesh[1][1][j] +=
            F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_F_dmesh[1][0][j] +=
            F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_F_dmesh[0][1][j] +=
            F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_F_dmesh[2][2][j] +=
              F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_F_dmesh[2][0][j] +=
              F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_F_dmesh[2][1][j] +=
              F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_F_dmesh[0][2][j] +=
              F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_F_dmesh[1][2][j] +=
              F_i * goma_ctx->fem.bf[v]->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif

  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][FILL] != -1) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_F_dmesh, 0, siz);
  }

  /*
   * d(grad(P))/dmesh
   */

  if (goma_ctx->fem.pd->gv[PRESSURE]) {
    v = PRESSURE;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NO_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_P_dmesh, 0, siz);
    for (i = 0; i < vdofs; i++) {
      P_i = *goma_ctx->fem.esp->P[i];
      for (p = 0; p < dimNonSym; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_P_dmesh[p][b][j] += P_i * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      P_i = *goma_ctx->fem.esp->P[0];
      goma_ctx->fem.fv->d_grad_P_dmesh[0][0][j] = P_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_P_dmesh[1][1][j] = P_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_P_dmesh[1][0][j] = P_i * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_P_dmesh[0][1][j] = P_i * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_P_dmesh[2][2][j] = P_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_P_dmesh[2][0][j] = P_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_P_dmesh[2][1][j] = P_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_P_dmesh[1][2][j] = P_i * bfv->d_grad_phi_dmesh[0][1][2][j];
        goma_ctx->fem.fv->d_grad_P_dmesh[0][2][j] = P_i * bfv->d_grad_phi_dmesh[0][0][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        P_i = *goma_ctx->fem.esp->P[i];
        goma_ctx->fem.fv->d_grad_P_dmesh[0][0][j] += P_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_P_dmesh[1][1][j] += P_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_P_dmesh[1][0][j] += P_i * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_P_dmesh[0][1][j] += P_i * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_P_dmesh[2][2][j] += P_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_P_dmesh[2][0][j] += P_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_P_dmesh[2][1][j] += P_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_P_dmesh[1][2][j] += P_i * bfv->d_grad_phi_dmesh[i][1][2][j];
          goma_ctx->fem.fv->d_grad_P_dmesh[0][2][j] += P_i * bfv->d_grad_phi_dmesh[i][0][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][PRESSURE] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_P_dmesh, 0, siz);
  }

  /*
   * d(grad(nn))/dmesh
   */

  if (goma_ctx->fem.pd->gv[BOND_EVOLUTION]) {
    v = BOND_EVOLUTION;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_nn_dmesh, 0, siz);
    for (i = 0; i < vdofs; i++) {
      for (p = 0; p < dim; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_nn_dmesh[p][b][j] +=
                *goma_ctx->fem.esp->nn[i] * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      goma_ctx->fem.fv->d_grad_nn_dmesh[0][0][j] =
          *goma_ctx->fem.esp->nn[0] * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_nn_dmesh[1][1][j] =
          *goma_ctx->fem.esp->nn[0] * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_nn_dmesh[1][0][j] =
          *goma_ctx->fem.esp->nn[0] * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_nn_dmesh[0][1][j] =
          *goma_ctx->fem.esp->nn[0] * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_nn_dmesh[2][2][j] =
            *goma_ctx->fem.esp->nn[0] * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_nn_dmesh[2][0][j] =
            *goma_ctx->fem.esp->nn[0] * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_nn_dmesh[2][1][j] =
            *goma_ctx->fem.esp->nn[0] * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_nn_dmesh[0][2][j] =
            *goma_ctx->fem.esp->nn[0] * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_nn_dmesh[1][2][j] =
            *goma_ctx->fem.esp->nn[0] * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        goma_ctx->fem.fv->d_grad_nn_dmesh[0][0][j] +=
            *goma_ctx->fem.esp->nn[i] * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_nn_dmesh[1][1][j] +=
            *goma_ctx->fem.esp->nn[i] * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_nn_dmesh[1][0][j] +=
            *goma_ctx->fem.esp->nn[i] * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_nn_dmesh[0][1][j] +=
            *goma_ctx->fem.esp->nn[i] * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_nn_dmesh[2][2][j] +=
              *goma_ctx->fem.esp->nn[i] * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_nn_dmesh[2][0][j] +=
              *goma_ctx->fem.esp->nn[i] * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_nn_dmesh[2][1][j] +=
              *goma_ctx->fem.esp->nn[i] * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_nn_dmesh[0][2][j] +=
              *goma_ctx->fem.esp->nn[i] * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_nn_dmesh[1][2][j] +=
              *goma_ctx->fem.esp->nn[i] * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  }

  /*
   * d(grad(c_w))/dmesh
   */
  if (goma_ctx->fem.pd->gv[MASS_FRACTION]) {
    v = MASS_FRACTION;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MAX_CONC * MDE;
    memset(goma_ctx->fem.fv->d_grad_c_dmesh, 0, siz);

    for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
      for (p = 0; p < dim; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            for (i = 0; i < vdofs; i++) {
              goma_ctx->fem.fv->d_grad_c_dmesh[p][w][b][j] +=
                  *goma_ctx->fem.esp->c[w][i] * bfv->d_grad_phi_dmesh[i][p][b][j];
            }
          }
        }
      }
    }
#else
    for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
      for (j = 0; j < mdofs; j++) {
        goma_ctx->fem.fv->d_grad_c_dmesh[0][w][0][j] =
            *goma_ctx->fem.esp->c[w][0] * bfv->d_grad_phi_dmesh[0][0][0][j];
        goma_ctx->fem.fv->d_grad_c_dmesh[1][w][1][j] =
            *goma_ctx->fem.esp->c[w][0] * bfv->d_grad_phi_dmesh[0][1][1][j];
        goma_ctx->fem.fv->d_grad_c_dmesh[1][w][0][j] =
            *goma_ctx->fem.esp->c[w][0] * bfv->d_grad_phi_dmesh[0][1][0][j];
        goma_ctx->fem.fv->d_grad_c_dmesh[0][w][1][j] =
            *goma_ctx->fem.esp->c[w][0] * bfv->d_grad_phi_dmesh[0][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_c_dmesh[2][w][2][j] =
              *goma_ctx->fem.esp->c[w][0] * bfv->d_grad_phi_dmesh[0][2][2][j];
          goma_ctx->fem.fv->d_grad_c_dmesh[2][w][0][j] =
              *goma_ctx->fem.esp->c[w][0] * bfv->d_grad_phi_dmesh[0][2][0][j];
          goma_ctx->fem.fv->d_grad_c_dmesh[2][w][1][j] =
              *goma_ctx->fem.esp->c[w][0] * bfv->d_grad_phi_dmesh[0][2][1][j];
          goma_ctx->fem.fv->d_grad_c_dmesh[0][w][2][j] =
              *goma_ctx->fem.esp->c[w][0] * bfv->d_grad_phi_dmesh[0][0][2][j];
          goma_ctx->fem.fv->d_grad_c_dmesh[1][w][2][j] =
              *goma_ctx->fem.esp->c[w][0] * bfv->d_grad_phi_dmesh[0][1][2][j];
        }

        for (i = 1; i < vdofs; i++) {
          goma_ctx->fem.fv->d_grad_c_dmesh[0][w][0][j] +=
              *goma_ctx->fem.esp->c[w][i] * bfv->d_grad_phi_dmesh[i][0][0][j];
          goma_ctx->fem.fv->d_grad_c_dmesh[1][w][1][j] +=
              *goma_ctx->fem.esp->c[w][i] * bfv->d_grad_phi_dmesh[i][1][1][j];
          goma_ctx->fem.fv->d_grad_c_dmesh[1][w][0][j] +=
              *goma_ctx->fem.esp->c[w][i] * bfv->d_grad_phi_dmesh[i][1][0][j];
          goma_ctx->fem.fv->d_grad_c_dmesh[0][w][1][j] +=
              *goma_ctx->fem.esp->c[w][i] * bfv->d_grad_phi_dmesh[i][0][1][j];

          if (dimNonSym == 3) {
            goma_ctx->fem.fv->d_grad_c_dmesh[2][w][2][j] +=
                *goma_ctx->fem.esp->c[w][i] * bfv->d_grad_phi_dmesh[i][2][2][j];
            goma_ctx->fem.fv->d_grad_c_dmesh[2][w][0][j] +=
                *goma_ctx->fem.esp->c[w][i] * bfv->d_grad_phi_dmesh[i][2][0][j];
            goma_ctx->fem.fv->d_grad_c_dmesh[2][w][1][j] +=
                *goma_ctx->fem.esp->c[w][i] * bfv->d_grad_phi_dmesh[i][2][1][j];
            goma_ctx->fem.fv->d_grad_c_dmesh[0][w][2][j] +=
                *goma_ctx->fem.esp->c[w][i] * bfv->d_grad_phi_dmesh[i][0][2][j];
            goma_ctx->fem.fv->d_grad_c_dmesh[1][w][2][j] +=
                *goma_ctx->fem.esp->c[w][i] * bfv->d_grad_phi_dmesh[i][1][2][j];
          }
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][MASS_FRACTION] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MAX_CONC * MDE;
    memset(goma_ctx->fem.fv->d_grad_c_dmesh, 0, siz);
  }

  /*
   * d(grad(porous_media_variables))/dmesh
   */
  if (goma_ctx->fem.pd->gv[POR_LIQ_PRES]) {
    v = POR_LIQ_PRES;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NO_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_p_liq_dmesh, 0, siz);

    for (p = 0; p < dim; p++) {
      for (b = 0; b < dim; b++) {
        for (j = 0; j < mdofs; j++) {
          for (i = 0; i < vdofs; i++) {
            goma_ctx->fem.fv->d_grad_p_liq_dmesh[p][b][j] +=
                *goma_ctx->fem.esp->p_liq[i] * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      P_i = *goma_ctx->fem.esp->p_liq[0];
      goma_ctx->fem.fv->d_grad_p_liq_dmesh[0][0][j] = P_i * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_p_liq_dmesh[1][1][j] = P_i * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_p_liq_dmesh[0][1][j] = P_i * bfv->d_grad_phi_dmesh[0][0][1][j];
      goma_ctx->fem.fv->d_grad_p_liq_dmesh[1][0][j] = P_i * bfv->d_grad_phi_dmesh[0][1][0][j];

      if (dim == 3) {
        goma_ctx->fem.fv->d_grad_p_liq_dmesh[2][2][j] = P_i * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_p_liq_dmesh[2][0][j] = P_i * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_p_liq_dmesh[2][1][j] = P_i * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_p_liq_dmesh[0][2][j] = P_i * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_p_liq_dmesh[1][2][j] = P_i * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        P_i = *goma_ctx->fem.esp->p_liq[i];
        /* fv->d_grad_p_liq_dmesh[p] [b][j] +=  P_i * bfv->d_grad_phi_dmesh[i][p] [b][j];*/
        goma_ctx->fem.fv->d_grad_p_liq_dmesh[0][0][j] += P_i * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_p_liq_dmesh[1][1][j] += P_i * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_p_liq_dmesh[0][1][j] += P_i * bfv->d_grad_phi_dmesh[i][0][1][j];
        goma_ctx->fem.fv->d_grad_p_liq_dmesh[1][0][j] += P_i * bfv->d_grad_phi_dmesh[i][1][0][j];

        if (dim == 3) {
          goma_ctx->fem.fv->d_grad_p_liq_dmesh[2][2][j] += P_i * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_p_liq_dmesh[2][0][j] += P_i * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_p_liq_dmesh[2][1][j] += P_i * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_p_liq_dmesh[0][2][j] += P_i * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_p_liq_dmesh[1][2][j] += P_i * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif

  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][POR_LIQ_PRES] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_p_liq_dmesh, 0, siz);
  }

  if (goma_ctx->fem.pd->gv[POR_GAS_PRES]) {
    v = POR_GAS_PRES;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_p_gas_dmesh, 0, siz);

    for (p = 0; p < dim; p++) {
      for (b = 0; b < dim; b++) {
        for (j = 0; j < mdofs; j++) {
          for (i = 0; i < vdofs; i++) {
            goma_ctx->fem.fv->d_grad_p_gas_dmesh[p][b][j] +=
                *goma_ctx->fem.esp->p_gas[i] * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      goma_ctx->fem.fv->d_grad_p_gas_dmesh[0][0][j] =
          *goma_ctx->fem.esp->p_gas[0] * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_p_gas_dmesh[1][1][j] =
          *goma_ctx->fem.esp->p_gas[0] * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_p_gas_dmesh[1][0][j] =
          *goma_ctx->fem.esp->p_gas[0] * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_p_gas_dmesh[0][1][j] =
          *goma_ctx->fem.esp->p_gas[0] * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_p_gas_dmesh[2][2][j] =
            *goma_ctx->fem.esp->p_gas[0] * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_p_gas_dmesh[2][0][j] =
            *goma_ctx->fem.esp->p_gas[0] * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_p_gas_dmesh[2][1][j] =
            *goma_ctx->fem.esp->p_gas[0] * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_p_gas_dmesh[0][2][j] =
            *goma_ctx->fem.esp->p_gas[0] * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_p_gas_dmesh[1][2][j] =
            *goma_ctx->fem.esp->p_gas[0] * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        goma_ctx->fem.fv->d_grad_p_gas_dmesh[0][0][j] +=
            *goma_ctx->fem.esp->p_gas[i] * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_p_gas_dmesh[1][1][j] +=
            *goma_ctx->fem.esp->p_gas[i] * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_p_gas_dmesh[1][0][j] +=
            *goma_ctx->fem.esp->p_gas[i] * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_p_gas_dmesh[0][1][j] +=
            *goma_ctx->fem.esp->p_gas[i] * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_p_gas_dmesh[2][2][j] +=
              *goma_ctx->fem.esp->p_gas[i] * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_p_gas_dmesh[2][0][j] +=
              *goma_ctx->fem.esp->p_gas[i] * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_p_gas_dmesh[2][1][j] +=
              *goma_ctx->fem.esp->p_gas[i] * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_p_gas_dmesh[0][2][j] +=
              *goma_ctx->fem.esp->p_gas[i] * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_p_gas_dmesh[1][2][j] +=
              *goma_ctx->fem.esp->p_gas[i] * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][POR_GAS_PRES] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_p_gas_dmesh, 0, siz);
  }

  if (goma_ctx->fem.pd->gv[POR_POROSITY]) {
    v = POR_POROSITY;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
#ifdef DO_NOT_UNROLL
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_porosity_dmesh, 0, siz);

    for (p = 0; p < dim; p++) {
      for (b = 0; b < dim; b++) {
        for (j = 0; j < mdofs; j++) {
          for (i = 0; i < vdofs; i++) {
            goma_ctx->fem.fv->d_grad_porosity_dmesh[p][b][j] +=
                *goma_ctx->fem.esp->porosity[i] * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
#else
    for (j = 0; j < mdofs; j++) {
      goma_ctx->fem.fv->d_grad_porosity_dmesh[0][0][j] =
          *goma_ctx->fem.esp->porosity[0] * bfv->d_grad_phi_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_grad_porosity_dmesh[1][1][j] =
          *goma_ctx->fem.esp->porosity[0] * bfv->d_grad_phi_dmesh[0][1][1][j];
      goma_ctx->fem.fv->d_grad_porosity_dmesh[1][0][j] =
          *goma_ctx->fem.esp->porosity[0] * bfv->d_grad_phi_dmesh[0][1][0][j];
      goma_ctx->fem.fv->d_grad_porosity_dmesh[0][1][j] =
          *goma_ctx->fem.esp->porosity[0] * bfv->d_grad_phi_dmesh[0][0][1][j];

      if (dimNonSym == 3) {
        goma_ctx->fem.fv->d_grad_porosity_dmesh[2][2][j] =
            *goma_ctx->fem.esp->porosity[0] * bfv->d_grad_phi_dmesh[0][2][2][j];
        goma_ctx->fem.fv->d_grad_porosity_dmesh[2][0][j] =
            *goma_ctx->fem.esp->porosity[0] * bfv->d_grad_phi_dmesh[0][2][0][j];
        goma_ctx->fem.fv->d_grad_porosity_dmesh[2][1][j] =
            *goma_ctx->fem.esp->porosity[0] * bfv->d_grad_phi_dmesh[0][2][1][j];
        goma_ctx->fem.fv->d_grad_porosity_dmesh[0][2][j] =
            *goma_ctx->fem.esp->porosity[0] * bfv->d_grad_phi_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_grad_porosity_dmesh[1][2][j] =
            *goma_ctx->fem.esp->porosity[0] * bfv->d_grad_phi_dmesh[0][1][2][j];
      }

      for (i = 1; i < vdofs; i++) {
        goma_ctx->fem.fv->d_grad_porosity_dmesh[0][0][j] +=
            *goma_ctx->fem.esp->porosity[i] * bfv->d_grad_phi_dmesh[i][0][0][j];
        goma_ctx->fem.fv->d_grad_porosity_dmesh[1][1][j] +=
            *goma_ctx->fem.esp->porosity[i] * bfv->d_grad_phi_dmesh[i][1][1][j];
        goma_ctx->fem.fv->d_grad_porosity_dmesh[1][0][j] +=
            *goma_ctx->fem.esp->porosity[i] * bfv->d_grad_phi_dmesh[i][1][0][j];
        goma_ctx->fem.fv->d_grad_porosity_dmesh[0][1][j] +=
            *goma_ctx->fem.esp->porosity[i] * bfv->d_grad_phi_dmesh[i][0][1][j];

        if (dimNonSym == 3) {
          goma_ctx->fem.fv->d_grad_porosity_dmesh[2][2][j] +=
              *goma_ctx->fem.esp->porosity[i] * bfv->d_grad_phi_dmesh[i][2][2][j];
          goma_ctx->fem.fv->d_grad_porosity_dmesh[2][0][j] +=
              *goma_ctx->fem.esp->porosity[i] * bfv->d_grad_phi_dmesh[i][2][0][j];
          goma_ctx->fem.fv->d_grad_porosity_dmesh[2][1][j] +=
              *goma_ctx->fem.esp->porosity[i] * bfv->d_grad_phi_dmesh[i][2][1][j];
          goma_ctx->fem.fv->d_grad_porosity_dmesh[0][2][j] +=
              *goma_ctx->fem.esp->porosity[i] * bfv->d_grad_phi_dmesh[i][0][2][j];
          goma_ctx->fem.fv->d_grad_porosity_dmesh[1][2][j] +=
              *goma_ctx->fem.esp->porosity[i] * bfv->d_grad_phi_dmesh[i][1][2][j];
        }
      }
    }
#endif
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][POR_POROSITY] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_porosity_dmesh, 0, siz);
  }

  /*
   * d( grad(v))/dmesh
   */
  if (goma_ctx->fem.pd->gv[VELOCITY1]) {
    v = VELOCITY1;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

#ifdef DO_NO_UNROLL
    siz = sizeof(double) * DIM * VIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_v_dmesh, 0, siz);

    for (r = 0; r < WIM; r++) {
      for (i = 0; i < vdofs; i++) {
        v_ri = *goma_ctx->fem.esp->v[r][i];
        for (p = 0; p < VIM; p++) {
          for (q = 0; q < VIM; q++) {
            for (b = 0; b < dim; b++) {
              for (j = 0; j < mdofs; j++) {
                goma_ctx->fem.fv->d_grad_v_dmesh[p][q][b][j] +=
                    v_ri * bfv->d_grad_phi_e_dmesh[i][r][p][q][b][j];
              }
            }
          }
        }
      }
    }
#else

    v_ri = *goma_ctx->fem.esp->v[0][0];

    for (b = 0; b < dim; b++) {
      for (j = 0; j < mdofs; j++) {
        goma_ctx->fem.fv->d_grad_v_dmesh[0][0][b][j] = 0.0;
        goma_ctx->fem.fv->d_grad_v_dmesh[1][1][b][j] = 0.0;
        goma_ctx->fem.fv->d_grad_v_dmesh[1][0][b][j] = 0.0;
        goma_ctx->fem.fv->d_grad_v_dmesh[0][1][b][j] = 0.0;
        if (VIMis3) {
          goma_ctx->fem.fv->d_grad_v_dmesh[2][2][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_v_dmesh[2][0][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_v_dmesh[2][1][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_v_dmesh[0][2][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_v_dmesh[1][2][b][j] = 0.0;
        }
      }
    }

    for (r = 0; r < WIM; r++) {
      for (i = 0; i < vdofs; i++) {
        v_ri = *goma_ctx->fem.esp->v[r][i];
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_v_dmesh[0][0][b][j] +=
                v_ri * bfv->d_grad_phi_e_dmesh[i][r][0][0][b][j];
            goma_ctx->fem.fv->d_grad_v_dmesh[1][1][b][j] +=
                v_ri * bfv->d_grad_phi_e_dmesh[i][r][1][1][b][j];
            goma_ctx->fem.fv->d_grad_v_dmesh[1][0][b][j] +=
                v_ri * bfv->d_grad_phi_e_dmesh[i][r][1][0][b][j];
            goma_ctx->fem.fv->d_grad_v_dmesh[0][1][b][j] +=
                v_ri * bfv->d_grad_phi_e_dmesh[i][r][0][1][b][j];

            if (VIMis3) {
              goma_ctx->fem.fv->d_grad_v_dmesh[2][2][b][j] +=
                  v_ri * bfv->d_grad_phi_e_dmesh[i][r][2][2][b][j];
              goma_ctx->fem.fv->d_grad_v_dmesh[2][0][b][j] +=
                  v_ri * bfv->d_grad_phi_e_dmesh[i][r][2][0][b][j];
              goma_ctx->fem.fv->d_grad_v_dmesh[2][1][b][j] +=
                  v_ri * bfv->d_grad_phi_e_dmesh[i][r][2][1][b][j];
              goma_ctx->fem.fv->d_grad_v_dmesh[0][2][b][j] +=
                  v_ri * bfv->d_grad_phi_e_dmesh[i][r][0][2][b][j];
              goma_ctx->fem.fv->d_grad_v_dmesh[1][2][b][j] +=
                  v_ri * bfv->d_grad_phi_e_dmesh[i][r][1][2][b][j];
            }
          }
        }
      }
    }
#endif
    /*
     * d( div(v) )/dmesh
     */
    /*
     * There is currently no need for div(pv), so this isn't cloned.
     */
    siz = sizeof(double) * DIM * MDE;
    memset(goma_ctx->fem.fv->d_div_v_dmesh, 0, siz);

    for (b = 0; b < dim; b++) {
      for (j = 0; j < mdofs; j++) {
        for (p = 0; p < VIM; p++) {
          goma_ctx->fem.fv->d_div_v_dmesh[b][j] += goma_ctx->fem.fv->d_grad_v_dmesh[p][p][b][j];
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][VELOCITY1] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_v_dmesh, 0, siz);
    siz = sizeof(double) * DIM * MDE;
    memset(goma_ctx->fem.fv->d_div_v_dmesh, 0, siz);
  }

  /*
   * d( grad(pv))/dmesh
   */
  if (goma_ctx->fem.pd->gv[PVELOCITY1]) {
    v = PVELOCITY1;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    siz = sizeof(double) * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_pv_dmesh, 0, siz);

    for (r = 0; r < WIM; r++) {
      for (i = 0; i < vdofs; i++) {
        v_ri = *goma_ctx->fem.esp->pv[r][i];
        for (p = 0; p < VIM; p++) {
          for (q = 0; q < VIM; q++) {
            for (b = 0; b < dim; b++) {
              for (j = 0; j < mdofs; j++) {
                goma_ctx->fem.fv->d_grad_pv_dmesh[p][q][b][j] +=
                    v_ri * bfv->d_grad_phi_e_dmesh[i][r][p][q][b][j];
              }
            }
          }
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][PVELOCITY1] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_pv_dmesh, 0, siz);
  }

  /*
   * d( grad(ext_v))/dmesh
   */
  if (goma_ctx->fem.pd->gv[EXT_VELOCITY]) {
    v = EXT_VELOCITY;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_ext_v_dmesh, 0, siz);

    for (i = 0; i < vdofs; i++) {
      v_ri = *goma_ctx->fem.esp->ext_v[i];
      for (p = 0; p < VIM; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_ext_v_dmesh[p][b][j] +=
                v_ri * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EXT_VELOCITY] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_ext_v_dmesh, 0, siz);
  }

  /*
   * d(grad(n))/dmesh  -- ls normal vector wrt mesh
   *
   * This is the gradient of the surface normal with respect to the mesh displacements.
   * This is a tensor quantity.
   *     d_grad_n_dmesh[p][q] [b][j]  -
   *
   *           p is the first coordinate of the tensor (nominally the derivative)
   *           q is the second cordinate of the tensor (nominally the direction of n)
   *           b is the mesh displacement coordinate.
   *           j is the mesh displacement degree of freedom in the current element
   *             (note for SHELL_NORMAL_1 this element refers to the shell element dof)
   *
   */
  if (goma_ctx->fem.pd->gv[NORMAL1] || goma_ctx->fem.pd->gv[SHELL_NORMAL1]) {
    if (goma_ctx->fem.pd->gv[NORMAL1])
      v = NORMAL1;
    if (goma_ctx->fem.pd->gv[SHELL_NORMAL1])
      v = SHELL_NORMAL1;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    siz = sizeof(double) * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_n_dmesh, 0, siz);
    siz = sizeof(double) * DIM * MDE;
    memset(goma_ctx->fem.fv->d_div_n_dmesh, 0, siz);

    if (v == NORMAL1) {
      // HKM -> looks good according to d_grad_v_dmesh[][][][]
      for (p = 0; p < VIM; p++) {
        for (q = 0; q < VIM; q++) {
          for (b = 0; b < dim; b++) {
            for (j = 0; j < mdofs; j++) {
              for (r = 0; r < dim; r++) {
                for (i = 0; i < vdofs; i++) {
                  v_ri = *goma_ctx->fem.esp->n[r][i];
                  goma_ctx->fem.fv->d_grad_n_dmesh[p][q][b][j] +=
                      v_ri * bfv->d_grad_phi_e_dmesh[i][r][p][q][b][j];
                }
              }
            }
          }
        }
      }
    }

    else if (v == SHELL_NORMAL1) {
      for (r = 0; r < dim; r++) {
        for (i = 0; i < vdofs; i++) {
          v_ri = *goma_ctx->fem.esp->n[r][i];
          for (p = 0; p < VIM; p++) {
            for (b = 0; b < dim; b++) {
              for (j = 0; j < mdofs; j++) {
                goma_ctx->fem.fv->d_grad_n_dmesh[p][r][b][j] +=
                    v_ri * bfv->d_grad_phi_dmesh[i][p][b][j];
              }
            }
          }
        }
      }
    }

    for (b = 0; b < dim; b++) {
      for (j = 0; j < mdofs; j++) {
        for (p = 0; p < VIM; p++) {
          goma_ctx->fem.fv->d_div_n_dmesh[b][j] += goma_ctx->fem.fv->d_grad_n_dmesh[p][p][b][j];
        }
      }
    }
  } else if (((goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][NORMAL1] != -1) &&
              (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_NORMAL1] != -1)) &&
             (okToZero)) {
    siz = sizeof(double) * DIM * VIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_n_dmesh, 0, siz);
    siz = sizeof(double) * DIM * MDE;
    memset(goma_ctx->fem.fv->d_div_n_dmesh, 0, siz);
  }

  /*
   * d( grad(E_field))/dmesh
   */

  if (goma_ctx->fem.pd->gv[EFIELD1]) {
    v = EFIELD1;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

    siz = sizeof(double) * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_E_field_dmesh, 0, siz);

    for (r = 0; r < WIM; r++) {
      for (i = 0; i < vdofs; i++) {
        v_ri = *goma_ctx->fem.esp->E_field[r][i];
        for (p = 0; p < VIM; p++) {
          for (q = 0; q < VIM; q++) {
            for (b = 0; b < dim; b++) {
              for (j = 0; j < mdofs; j++) {
                goma_ctx->fem.fv->d_grad_E_field_dmesh[p][q][b][j] +=
                    v_ri * bfv->d_grad_phi_e_dmesh[i][r][p][q][b][j];
              }
            }
          }
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EFIELD1] != -1 && okToZero) {
    siz = sizeof(double) * DIM * VIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_E_field_dmesh, 0, siz);
  }

  /*
   * d( grad(em_er))/dmesh
   */
  if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][EM_E1_REAL] ||
      goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][EM_E2_REAL] ||
      goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][EM_E3_REAL]) {
    v = EM_E1_REAL;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];

#ifdef DO_NO_UNROLL
    siz = sizeof(double) * DIM * VIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_em_er_dmesh, 0, siz);

    for (r = 0; r < WIM; r++) {
      for (i = 0; i < vdofs; i++) {
        em_er_ri = *goma_ctx->fem.esp->em_er[r][i];
        for (p = 0; p < VIM; p++) {
          for (q = 0; q < VIM; q++) {
            for (b = 0; b < dim; b++) {
              for (j = 0; j < mdofs; j++) {
                goma_ctx->fem.fv->d_grad_em_er_dmesh[p][q][b][j] +=
                    em_er_ri * bfv->d_grad_phi_e_dmesh[i][r][p][q][b][j];
              }
            }
          }
        }
      }
    }
#else

    em_er_ri = *goma_ctx->fem.esp->em_er[0][0];

    for (b = 0; b < dim; b++) {
      for (j = 0; j < mdofs; j++) {
        goma_ctx->fem.fv->d_grad_em_er_dmesh[0][0][b][j] = 0.0;
        goma_ctx->fem.fv->d_grad_em_er_dmesh[1][1][b][j] = 0.0;
        goma_ctx->fem.fv->d_grad_em_er_dmesh[1][0][b][j] = 0.0;
        goma_ctx->fem.fv->d_grad_em_er_dmesh[0][1][b][j] = 0.0;
        if (VIMis3) {
          goma_ctx->fem.fv->d_grad_em_er_dmesh[2][2][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_em_er_dmesh[2][0][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_em_er_dmesh[2][1][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_em_er_dmesh[0][2][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_em_er_dmesh[1][2][b][j] = 0.0;
        }
      }
    }

    for (r = 0; r < WIM; r++) {
      for (i = 0; i < vdofs; i++) {
        em_er_ri = *goma_ctx->fem.esp->em_er[r][i];
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_em_er_dmesh[0][0][b][j] +=
                em_er_ri * bfv->d_grad_phi_e_dmesh[i][r][0][0][b][j];
            goma_ctx->fem.fv->d_grad_em_er_dmesh[1][1][b][j] +=
                em_er_ri * bfv->d_grad_phi_e_dmesh[i][r][1][1][b][j];
            goma_ctx->fem.fv->d_grad_em_er_dmesh[1][0][b][j] +=
                em_er_ri * bfv->d_grad_phi_e_dmesh[i][r][1][0][b][j];
            goma_ctx->fem.fv->d_grad_em_er_dmesh[0][1][b][j] +=
                em_er_ri * bfv->d_grad_phi_e_dmesh[i][r][0][1][b][j];

            if (VIMis3) {
              goma_ctx->fem.fv->d_grad_em_er_dmesh[2][2][b][j] +=
                  em_er_ri * bfv->d_grad_phi_e_dmesh[i][r][2][2][b][j];
              goma_ctx->fem.fv->d_grad_em_er_dmesh[2][0][b][j] +=
                  em_er_ri * bfv->d_grad_phi_e_dmesh[i][r][2][0][b][j];
              goma_ctx->fem.fv->d_grad_em_er_dmesh[2][1][b][j] +=
                  em_er_ri * bfv->d_grad_phi_e_dmesh[i][r][2][1][b][j];
              goma_ctx->fem.fv->d_grad_em_er_dmesh[0][2][b][j] +=
                  em_er_ri * bfv->d_grad_phi_e_dmesh[i][r][0][2][b][j];
              goma_ctx->fem.fv->d_grad_em_er_dmesh[1][2][b][j] +=
                  em_er_ri * bfv->d_grad_phi_e_dmesh[i][r][1][2][b][j];
            }
          }
        }
      }
    }
#endif

  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_E1_REAL] == -1 &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_E2_REAL] == -1 &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_E3_REAL] == -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_em_er_dmesh, 0, siz);
  }

  /*
   * d( grad(em_ei))/dmesh
   */
  if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][EM_E1_IMAG] ||
      goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][EM_E2_IMAG] ||
      goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][EM_E3_IMAG]) {
    v = EM_E1_IMAG;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];

#ifdef DO_NO_UNROLL
    siz = sizeof(double) * DIM * VIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_em_ei_dmesh, 0, siz);

    for (r = 0; r < WIM; r++) {
      for (i = 0; i < vdofs; i++) {
        em_ei_ri = *goma_ctx->fem.esp->em_ei[r][i];
        for (p = 0; p < VIM; p++) {
          for (q = 0; q < VIM; q++) {
            for (b = 0; b < dim; b++) {
              for (j = 0; j < mdofs; j++) {
                goma_ctx->fem.fv->d_grad_em_ei_dmesh[p][q][b][j] +=
                    em_ei_ri * bfv->d_grad_phi_e_dmesh[i][r][p][q][b][j];
              }
            }
          }
        }
      }
    }
#else

    em_ei_ri = *goma_ctx->fem.esp->em_ei[0][0];

    for (b = 0; b < dim; b++) {
      for (j = 0; j < mdofs; j++) {
        goma_ctx->fem.fv->d_grad_em_ei_dmesh[0][0][b][j] = 0.0;
        goma_ctx->fem.fv->d_grad_em_ei_dmesh[1][1][b][j] = 0.0;
        goma_ctx->fem.fv->d_grad_em_ei_dmesh[1][0][b][j] = 0.0;
        goma_ctx->fem.fv->d_grad_em_ei_dmesh[0][1][b][j] = 0.0;
        if (VIMis3) {
          goma_ctx->fem.fv->d_grad_em_ei_dmesh[2][2][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_em_ei_dmesh[2][0][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_em_ei_dmesh[2][1][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_em_ei_dmesh[0][2][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_em_ei_dmesh[1][2][b][j] = 0.0;
        }
      }
    }

    for (r = 0; r < WIM; r++) {
      for (i = 0; i < vdofs; i++) {
        em_ei_ri = *goma_ctx->fem.esp->em_ei[r][i];
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_em_ei_dmesh[0][0][b][j] +=
                em_ei_ri * bfv->d_grad_phi_e_dmesh[i][r][0][0][b][j];
            goma_ctx->fem.fv->d_grad_em_ei_dmesh[1][1][b][j] +=
                em_ei_ri * bfv->d_grad_phi_e_dmesh[i][r][1][1][b][j];
            goma_ctx->fem.fv->d_grad_em_ei_dmesh[1][0][b][j] +=
                em_ei_ri * bfv->d_grad_phi_e_dmesh[i][r][1][0][b][j];
            goma_ctx->fem.fv->d_grad_em_ei_dmesh[0][1][b][j] +=
                em_ei_ri * bfv->d_grad_phi_e_dmesh[i][r][0][1][b][j];

            if (VIMis3) {
              goma_ctx->fem.fv->d_grad_em_ei_dmesh[2][2][b][j] +=
                  em_ei_ri * bfv->d_grad_phi_e_dmesh[i][r][2][2][b][j];
              goma_ctx->fem.fv->d_grad_em_ei_dmesh[2][0][b][j] +=
                  em_ei_ri * bfv->d_grad_phi_e_dmesh[i][r][2][0][b][j];
              goma_ctx->fem.fv->d_grad_em_ei_dmesh[2][1][b][j] +=
                  em_ei_ri * bfv->d_grad_phi_e_dmesh[i][r][2][1][b][j];
              goma_ctx->fem.fv->d_grad_em_ei_dmesh[0][2][b][j] +=
                  em_ei_ri * bfv->d_grad_phi_e_dmesh[i][r][0][2][b][j];
              goma_ctx->fem.fv->d_grad_em_ei_dmesh[1][2][b][j] +=
                  em_ei_ri * bfv->d_grad_phi_e_dmesh[i][r][1][2][b][j];
            }
          }
        }
      }
    }
#endif

  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_E1_IMAG] == -1 &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_E2_IMAG] == -1 &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_E3_IMAG] == -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_em_ei_dmesh, 0, siz);
  }

  /*
   * d( grad(em_hr))/dmesh
   */
  if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][EM_H1_REAL] ||
      goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][EM_H2_REAL] ||
      goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][EM_H3_REAL]) {
    v = EM_H1_REAL;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];

#ifdef DO_NO_UNROLL
    siz = sizeof(double) * DIM * VIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_em_hr_dmesh, 0, siz);

    for (r = 0; r < WIM; r++) {
      for (i = 0; i < vdofs; i++) {
        em_hr_ri = *goma_ctx->fem.esp->em_hr[r][i];
        for (p = 0; p < VIM; p++) {
          for (q = 0; q < VIM; q++) {
            for (b = 0; b < dim; b++) {
              for (j = 0; j < mdofs; j++) {
                goma_ctx->fem.fv->d_grad_em_hr_dmesh[p][q][b][j] +=
                    em_hr_ri * bfv->d_grad_phi_e_dmesh[i][r][p][q][b][j];
              }
            }
          }
        }
      }
    }
#else

    em_hr_ri = *goma_ctx->fem.esp->em_hr[0][0];

    for (b = 0; b < dim; b++) {
      for (j = 0; j < mdofs; j++) {
        goma_ctx->fem.fv->d_grad_em_hr_dmesh[0][0][b][j] = 0.0;
        goma_ctx->fem.fv->d_grad_em_hr_dmesh[1][1][b][j] = 0.0;
        goma_ctx->fem.fv->d_grad_em_hr_dmesh[1][0][b][j] = 0.0;
        goma_ctx->fem.fv->d_grad_em_hr_dmesh[0][1][b][j] = 0.0;
        if (VIMis3) {
          goma_ctx->fem.fv->d_grad_em_hr_dmesh[2][2][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_em_hr_dmesh[2][0][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_em_hr_dmesh[2][1][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_em_hr_dmesh[0][2][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_em_hr_dmesh[1][2][b][j] = 0.0;
        }
      }
    }

    for (r = 0; r < WIM; r++) {
      for (i = 0; i < vdofs; i++) {
        em_hr_ri = *goma_ctx->fem.esp->em_hr[r][i];
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_em_hr_dmesh[0][0][b][j] +=
                em_hr_ri * bfv->d_grad_phi_e_dmesh[i][r][0][0][b][j];
            goma_ctx->fem.fv->d_grad_em_hr_dmesh[1][1][b][j] +=
                em_hr_ri * bfv->d_grad_phi_e_dmesh[i][r][1][1][b][j];
            goma_ctx->fem.fv->d_grad_em_hr_dmesh[1][0][b][j] +=
                em_hr_ri * bfv->d_grad_phi_e_dmesh[i][r][1][0][b][j];
            goma_ctx->fem.fv->d_grad_em_hr_dmesh[0][1][b][j] +=
                em_hr_ri * bfv->d_grad_phi_e_dmesh[i][r][0][1][b][j];

            if (VIMis3) {
              goma_ctx->fem.fv->d_grad_em_hr_dmesh[2][2][b][j] +=
                  em_hr_ri * bfv->d_grad_phi_e_dmesh[i][r][2][2][b][j];
              goma_ctx->fem.fv->d_grad_em_hr_dmesh[2][0][b][j] +=
                  em_hr_ri * bfv->d_grad_phi_e_dmesh[i][r][2][0][b][j];
              goma_ctx->fem.fv->d_grad_em_hr_dmesh[2][1][b][j] +=
                  em_hr_ri * bfv->d_grad_phi_e_dmesh[i][r][2][1][b][j];
              goma_ctx->fem.fv->d_grad_em_hr_dmesh[0][2][b][j] +=
                  em_hr_ri * bfv->d_grad_phi_e_dmesh[i][r][0][2][b][j];
              goma_ctx->fem.fv->d_grad_em_hr_dmesh[1][2][b][j] +=
                  em_hr_ri * bfv->d_grad_phi_e_dmesh[i][r][1][2][b][j];
            }
          }
        }
      }
    }
#endif

  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_H1_REAL] == -1 &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_H2_REAL] == -1 &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_H3_REAL] == -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_em_hr_dmesh, 0, siz);
  }

  /*
   * d( grad(em_hi))/dmesh
   */
  if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][EM_H1_IMAG] ||
      goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][EM_H2_IMAG] ||
      goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][EM_H3_IMAG]) {
    v = EM_H1_IMAG;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[v];

#ifdef DO_NO_UNROLL
    siz = sizeof(double) * DIM * VIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_em_hi_dmesh, 0, siz);

    for (r = 0; r < WIM; r++) {
      for (i = 0; i < vdofs; i++) {
        em_hi_ri = *goma_ctx->fem.esp->em_hi[r][i];
        for (p = 0; p < VIM; p++) {
          for (q = 0; q < VIM; q++) {
            for (b = 0; b < dim; b++) {
              for (j = 0; j < mdofs; j++) {
                goma_ctx->fem.fv->d_grad_em_hi_dmesh[p][q][b][j] +=
                    em_hi_ri * bfv->d_grad_phi_e_dmesh[i][r][p][q][b][j];
              }
            }
          }
        }
      }
    }
#else

    em_hi_ri = *goma_ctx->fem.esp->em_hi[0][0];

    for (b = 0; b < dim; b++) {
      for (j = 0; j < mdofs; j++) {
        goma_ctx->fem.fv->d_grad_em_hi_dmesh[0][0][b][j] = 0.0;
        goma_ctx->fem.fv->d_grad_em_hi_dmesh[1][1][b][j] = 0.0;
        goma_ctx->fem.fv->d_grad_em_hi_dmesh[1][0][b][j] = 0.0;
        goma_ctx->fem.fv->d_grad_em_hi_dmesh[0][1][b][j] = 0.0;
        if (VIMis3) {
          goma_ctx->fem.fv->d_grad_em_hi_dmesh[2][2][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_em_hi_dmesh[2][0][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_em_hi_dmesh[2][1][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_em_hi_dmesh[0][2][b][j] = 0.0;
          goma_ctx->fem.fv->d_grad_em_hi_dmesh[1][2][b][j] = 0.0;
        }
      }
    }

    for (r = 0; r < WIM; r++) {
      for (i = 0; i < vdofs; i++) {
        em_hi_ri = *goma_ctx->fem.esp->em_hi[r][i];
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_grad_em_hi_dmesh[0][0][b][j] +=
                em_hi_ri * bfv->d_grad_phi_e_dmesh[i][r][0][0][b][j];
            goma_ctx->fem.fv->d_grad_em_hi_dmesh[1][1][b][j] +=
                em_hi_ri * bfv->d_grad_phi_e_dmesh[i][r][1][1][b][j];
            goma_ctx->fem.fv->d_grad_em_hi_dmesh[1][0][b][j] +=
                em_hi_ri * bfv->d_grad_phi_e_dmesh[i][r][1][0][b][j];
            goma_ctx->fem.fv->d_grad_em_hi_dmesh[0][1][b][j] +=
                em_hi_ri * bfv->d_grad_phi_e_dmesh[i][r][0][1][b][j];

            if (VIMis3) {
              goma_ctx->fem.fv->d_grad_em_hi_dmesh[2][2][b][j] +=
                  em_hi_ri * bfv->d_grad_phi_e_dmesh[i][r][2][2][b][j];
              goma_ctx->fem.fv->d_grad_em_hi_dmesh[2][0][b][j] +=
                  em_hi_ri * bfv->d_grad_phi_e_dmesh[i][r][2][0][b][j];
              goma_ctx->fem.fv->d_grad_em_hi_dmesh[2][1][b][j] +=
                  em_hi_ri * bfv->d_grad_phi_e_dmesh[i][r][2][1][b][j];
              goma_ctx->fem.fv->d_grad_em_hi_dmesh[0][2][b][j] +=
                  em_hi_ri * bfv->d_grad_phi_e_dmesh[i][r][0][2][b][j];
              goma_ctx->fem.fv->d_grad_em_hi_dmesh[1][2][b][j] +=
                  em_hi_ri * bfv->d_grad_phi_e_dmesh[i][r][1][2][b][j];
            }
          }
        }
      }
    }
#endif

  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_H1_IMAG] == -1 &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_H2_IMAG] == -1 &&
             goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][EM_H3_IMAG] == -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_em_hi_dmesh, 0, siz);
  }
  /*
   * d( grad(d) )/dmesh
   * d( grad(d_dot)/dmesh
   */

  /* Yes, I know it should be each "+p", */
  /* but this way we can collect three */
  /* components of vectors and tensors for */
  /* less than 3 independent spatial dimensions*/

  if (goma_ctx->fem.pd->gv[MESH_DISPLACEMENT1] == 1) {
    v = MESH_DISPLACEMENT1;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

#ifdef DO_NO_UNROLL
    siz = sizeof(double) * DIM * MDE;
    memset(goma_ctx->fem.fv->d_div_d_dmesh, 0, siz);
    siz = sizeof(double) * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_d_dmesh, 0, siz);
    memset(goma_ctx->fem.fv->d_grad_d_dot_dmesh, 0, siz);
    for (r = 0; r < dim; r++) {
      for (i = 0; i < vdofs; i++) {
        d_ri = *goma_ctx->fem.esp->d[r][i];
        for (p = 0; p < VIM; p++) {
          for (q = 0; q < VIM; q++) {
            for (b = 0; b < dim; b++) {
              for (j = 0; j < mdofs; j++) {
                /*  fv->d_grad_d_dmesh[p][q] [b][j] = 0.; */
                goma_ctx->fem.fv->d_grad_d_dmesh[p][q][b][j] +=
                    d_ri * bfv->d_grad_phi_e_dmesh[i][r][p][q][b][j];

                goma_ctx->fem.fv->d_grad_d_dot_dmesh[p][q][b][j] +=
                    *goma_ctx->fem.esp_dot->d[r][i] * bfv->d_grad_phi_e_dmesh[i][r][p][q][b][j];
              }
            }
          }
        }
      }
    }
#else

    d_ri = *goma_ctx->fem.esp->d[0][0];
    d_dot_ri = *goma_ctx->fem.esp_dot->d[0][0];

    for (b = 0; b < dim; b++) {
      for (j = 0; j < mdofs; j++) {
        /* The first element of the (r,i) matrix is split out so that it can be initialized by
         * direct assignment This saves having to keep doing memsets on d_grad_d_mesh */

        /*	  fv->d_grad_d_dmesh[p][q] [b][j] += d_ri * bfv->d_grad_phi_e_dmesh[i][r] [p][q]
         * [b][j]; */

        goma_ctx->fem.fv->d_grad_d_dmesh[0][0][b][j] =
            d_ri * bfv->d_grad_phi_e_dmesh[0][0][0][0][b][j];
        goma_ctx->fem.fv->d_grad_d_dmesh[1][1][b][j] =
            d_ri * bfv->d_grad_phi_e_dmesh[0][0][1][1][b][j];
        goma_ctx->fem.fv->d_grad_d_dmesh[1][0][b][j] =
            d_ri * bfv->d_grad_phi_e_dmesh[0][0][1][0][b][j];
        goma_ctx->fem.fv->d_grad_d_dmesh[0][1][b][j] =
            d_ri * bfv->d_grad_phi_e_dmesh[0][0][0][1][b][j];

        /*  fv->d_grad_d_dot_dmesh[p][q] [b][j] += d_dot_ri * bfv->d_grad_phi_e_dmesh[i][r] [p][q]
         * [b][j];*/

        goma_ctx->fem.fv->d_grad_d_dot_dmesh[0][0][b][j] =
            d_dot_ri * bfv->d_grad_phi_e_dmesh[0][0][0][0][b][j];
        goma_ctx->fem.fv->d_grad_d_dot_dmesh[1][1][b][j] =
            d_dot_ri * bfv->d_grad_phi_e_dmesh[0][0][1][1][b][j];
        goma_ctx->fem.fv->d_grad_d_dot_dmesh[1][0][b][j] =
            d_dot_ri * bfv->d_grad_phi_e_dmesh[0][0][1][0][b][j];
        goma_ctx->fem.fv->d_grad_d_dot_dmesh[0][1][b][j] =
            d_dot_ri * bfv->d_grad_phi_e_dmesh[0][0][0][1][b][j];

        if (VIMis3) {
          goma_ctx->fem.fv->d_grad_d_dmesh[2][2][b][j] =
              d_ri * bfv->d_grad_phi_e_dmesh[0][0][2][2][b][j];
          goma_ctx->fem.fv->d_grad_d_dmesh[2][0][b][j] =
              d_ri * bfv->d_grad_phi_e_dmesh[0][0][2][0][b][j];
          goma_ctx->fem.fv->d_grad_d_dmesh[2][1][b][j] =
              d_ri * bfv->d_grad_phi_e_dmesh[0][0][2][1][b][j];
          goma_ctx->fem.fv->d_grad_d_dmesh[0][2][b][j] =
              d_ri * bfv->d_grad_phi_e_dmesh[0][0][0][2][b][j];
          goma_ctx->fem.fv->d_grad_d_dmesh[1][2][b][j] =
              d_ri * bfv->d_grad_phi_e_dmesh[0][0][1][2][b][j];

          goma_ctx->fem.fv->d_grad_d_dot_dmesh[2][2][b][j] =
              d_dot_ri * bfv->d_grad_phi_e_dmesh[0][0][2][2][b][j];
          goma_ctx->fem.fv->d_grad_d_dot_dmesh[2][0][b][j] =
              d_dot_ri * bfv->d_grad_phi_e_dmesh[0][0][2][0][b][j];
          goma_ctx->fem.fv->d_grad_d_dot_dmesh[2][1][b][j] =
              d_dot_ri * bfv->d_grad_phi_e_dmesh[0][0][2][1][b][j];
          goma_ctx->fem.fv->d_grad_d_dot_dmesh[0][2][b][j] =
              d_dot_ri * bfv->d_grad_phi_e_dmesh[0][0][0][2][b][j];
          goma_ctx->fem.fv->d_grad_d_dot_dmesh[1][2][b][j] =
              d_dot_ri * bfv->d_grad_phi_e_dmesh[0][0][1][2][b][j];
        }
      }
    }

    for (r = 0; r < dim; r++) {
      for (i = 0; i < vdofs; i++) {
        d_ri = *goma_ctx->fem.esp->d[r][i];
        d_dot_ri = *goma_ctx->fem.esp_dot->d[r][i];

        for (b = 0; (r + i) && (b < dim);
             b++) /* the (r+i) test is needed to make sure only the (r=0,i=0) element is excluded */
        {
          for (j = 0; j < mdofs; j++) {

            /*	  fv->d_grad_d_dmesh[p][q] [b][j] += d_ri * bfv->d_grad_phi_e_dmesh[i][r] [p][q]
             * [b][j]; */

            goma_ctx->fem.fv->d_grad_d_dmesh[0][0][b][j] +=
                d_ri * bfv->d_grad_phi_e_dmesh[i][r][0][0][b][j];
            goma_ctx->fem.fv->d_grad_d_dmesh[1][1][b][j] +=
                d_ri * bfv->d_grad_phi_e_dmesh[i][r][1][1][b][j];
            goma_ctx->fem.fv->d_grad_d_dmesh[1][0][b][j] +=
                d_ri * bfv->d_grad_phi_e_dmesh[i][r][1][0][b][j];
            goma_ctx->fem.fv->d_grad_d_dmesh[0][1][b][j] +=
                d_ri * bfv->d_grad_phi_e_dmesh[i][r][0][1][b][j];

            /*  fv->d_grad_d_dot_dmesh[p][q] [b][j] += d_dot_ri * bfv->d_grad_phi_e_dmesh[i][r]
             * [p][q] [b][j];*/

            goma_ctx->fem.fv->d_grad_d_dot_dmesh[0][0][b][j] +=
                d_dot_ri * bfv->d_grad_phi_e_dmesh[i][r][0][0][b][j];
            goma_ctx->fem.fv->d_grad_d_dot_dmesh[1][1][b][j] +=
                d_dot_ri * bfv->d_grad_phi_e_dmesh[i][r][1][1][b][j];
            goma_ctx->fem.fv->d_grad_d_dot_dmesh[1][0][b][j] +=
                d_dot_ri * bfv->d_grad_phi_e_dmesh[i][r][1][0][b][j];
            goma_ctx->fem.fv->d_grad_d_dot_dmesh[0][1][b][j] +=
                d_dot_ri * bfv->d_grad_phi_e_dmesh[i][r][0][1][b][j];

            if (VIMis3) {
              goma_ctx->fem.fv->d_grad_d_dmesh[2][2][b][j] +=
                  d_ri * bfv->d_grad_phi_e_dmesh[i][r][2][2][b][j];
              goma_ctx->fem.fv->d_grad_d_dmesh[2][0][b][j] +=
                  d_ri * bfv->d_grad_phi_e_dmesh[i][r][2][0][b][j];
              goma_ctx->fem.fv->d_grad_d_dmesh[2][1][b][j] +=
                  d_ri * bfv->d_grad_phi_e_dmesh[i][r][2][1][b][j];
              goma_ctx->fem.fv->d_grad_d_dmesh[0][2][b][j] +=
                  d_ri * bfv->d_grad_phi_e_dmesh[i][r][0][2][b][j];
              goma_ctx->fem.fv->d_grad_d_dmesh[1][2][b][j] +=
                  d_ri * bfv->d_grad_phi_e_dmesh[i][r][1][2][b][j];

              goma_ctx->fem.fv->d_grad_d_dot_dmesh[2][2][b][j] +=
                  d_dot_ri * bfv->d_grad_phi_e_dmesh[i][r][2][2][b][j];
              goma_ctx->fem.fv->d_grad_d_dot_dmesh[2][0][b][j] +=
                  d_dot_ri * bfv->d_grad_phi_e_dmesh[i][r][2][0][b][j];
              goma_ctx->fem.fv->d_grad_d_dot_dmesh[2][1][b][j] +=
                  d_dot_ri * bfv->d_grad_phi_e_dmesh[i][r][2][1][b][j];
              goma_ctx->fem.fv->d_grad_d_dot_dmesh[0][2][b][j] +=
                  d_dot_ri * bfv->d_grad_phi_e_dmesh[i][r][0][2][b][j];
              goma_ctx->fem.fv->d_grad_d_dot_dmesh[1][2][b][j] +=
                  d_dot_ri * bfv->d_grad_phi_e_dmesh[i][r][1][2][b][j];
            }
          }
        }
      }
    }
#endif

#ifdef DO_NO_UNROLL
    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            /*
             * Special added Lagrangian piece for d grad_d / d mesh...
             */
            goma_ctx->fem.fv->d_grad_d_dmesh[p][q][b][j] += bfv->grad_phi_e[j][b][p][q];

            /*N.B. PRS: need to actually furbish this for Newmark Beta schemes */
            if (transient_run && (discontinuous_porous_media || ve_solid)) {
              goma_ctx->fem.fv->d_grad_d_dot_dmesh[p][q][b][j] +=
                  (1. + 2. * goma_ctx->fem.tran->theta) * bfv->grad_phi_e[j][b][p][q] /
                  goma_ctx->fem.tran->delta_t;
            }
          }
        }
      }
    }
#else

    for (b = 0; b < dim; b++) {
      for (j = 0; j < mdofs; j++) {
        /*
         * Special added Lagrangian piece for d grad_d / d mesh...
         */
        /*  fv->d_grad_d_dmesh[p][q] [b][j] += bfv->grad_phi_e[j][b] [p][q];*/

        goma_ctx->fem.fv->d_grad_d_dmesh[0][0][b][j] += bfv->grad_phi_e[j][b][0][0];
        goma_ctx->fem.fv->d_grad_d_dmesh[1][1][b][j] += bfv->grad_phi_e[j][b][1][1];
        goma_ctx->fem.fv->d_grad_d_dmesh[1][0][b][j] += bfv->grad_phi_e[j][b][1][0];
        goma_ctx->fem.fv->d_grad_d_dmesh[0][1][b][j] += bfv->grad_phi_e[j][b][0][1];

        if (VIMis3) {
          goma_ctx->fem.fv->d_grad_d_dmesh[2][2][b][j] += bfv->grad_phi_e[j][b][2][2];
          goma_ctx->fem.fv->d_grad_d_dmesh[2][0][b][j] += bfv->grad_phi_e[j][b][2][0];
          goma_ctx->fem.fv->d_grad_d_dmesh[2][1][b][j] += bfv->grad_phi_e[j][b][2][1];
          goma_ctx->fem.fv->d_grad_d_dmesh[0][2][b][j] += bfv->grad_phi_e[j][b][0][2];
          goma_ctx->fem.fv->d_grad_d_dmesh[1][2][b][j] += bfv->grad_phi_e[j][b][1][2];
        }

        /*N.B. PRS: need to actually furbish this for Newmark Beta schemes */
        if (transient_run && (discontinuous_porous_media || ve_solid)) {
          /* fv->d_grad_d_dot_dmesh[p][q] [b][j] += (1.+2.*tran->theta)*bfv->grad_phi_e[j][b]
           * [p][q]/tran->delta_t;*/

          goma_ctx->fem.fv->d_grad_d_dot_dmesh[0][0][b][j] +=
              (1. + 2. * goma_ctx->fem.tran->theta) * bfv->grad_phi_e[j][b][0][0] /
              goma_ctx->fem.tran->delta_t;
          goma_ctx->fem.fv->d_grad_d_dot_dmesh[1][1][b][j] +=
              (1. + 2. * goma_ctx->fem.tran->theta) * bfv->grad_phi_e[j][b][1][1] /
              goma_ctx->fem.tran->delta_t;
          goma_ctx->fem.fv->d_grad_d_dot_dmesh[1][0][b][j] +=
              (1. + 2. * goma_ctx->fem.tran->theta) * bfv->grad_phi_e[j][b][1][0] /
              goma_ctx->fem.tran->delta_t;
          goma_ctx->fem.fv->d_grad_d_dot_dmesh[0][1][b][j] +=
              (1. + 2. * goma_ctx->fem.tran->theta) * bfv->grad_phi_e[j][b][0][1] /
              goma_ctx->fem.tran->delta_t;

          if (VIMis3) {

            goma_ctx->fem.fv->d_grad_d_dot_dmesh[2][2][b][j] +=
                (1. + 2. * goma_ctx->fem.tran->theta) * bfv->grad_phi_e[j][b][2][2] /
                goma_ctx->fem.tran->delta_t;
            goma_ctx->fem.fv->d_grad_d_dot_dmesh[2][0][b][j] +=
                (1. + 2. * goma_ctx->fem.tran->theta) * bfv->grad_phi_e[j][b][2][0] /
                goma_ctx->fem.tran->delta_t;
            goma_ctx->fem.fv->d_grad_d_dot_dmesh[2][1][b][j] +=
                (1. + 2. * goma_ctx->fem.tran->theta) * bfv->grad_phi_e[j][b][2][1] /
                goma_ctx->fem.tran->delta_t;
            goma_ctx->fem.fv->d_grad_d_dot_dmesh[0][2][b][j] +=
                (1. + 2. * goma_ctx->fem.tran->theta) * bfv->grad_phi_e[j][b][0][2] /
                goma_ctx->fem.tran->delta_t;
            goma_ctx->fem.fv->d_grad_d_dot_dmesh[1][2][b][j] +=
                (1. + 2. * goma_ctx->fem.tran->theta) * bfv->grad_phi_e[j][b][1][2] /
                goma_ctx->fem.tran->delta_t;
          }
        }
      }
    }

#endif

    /*
     * d( div(d) )/dmesh [ just trace(grad(d)) ]
     */
#ifdef DO_NO_UNROLL
    for (b = 0; b < dim; b++) {
      for (j = 0; j < mdofs; j++) {
        for (p = 0; p < VIM; p++) {
          goma_ctx->fem.fv->d_div_d_dmesh[b][j] += goma_ctx->fem.fv->d_grad_d_dmesh[p][p][b][j];
        }
      }
    }
#else

    goma_ctx->fem.fv->d_div_d_dmesh[0][0] = goma_ctx->fem.fv->d_grad_d_dmesh[0][0][0][0];
    goma_ctx->fem.fv->d_div_d_dmesh[1][0] = goma_ctx->fem.fv->d_grad_d_dmesh[1][1][1][0];
    goma_ctx->fem.fv->d_div_d_dmesh[0][0] = goma_ctx->fem.fv->d_grad_d_dmesh[1][1][0][0];
    goma_ctx->fem.fv->d_div_d_dmesh[1][0] = goma_ctx->fem.fv->d_grad_d_dmesh[0][0][1][0];

    if (VIMis3) {
      goma_ctx->fem.fv->d_div_d_dmesh[2][0] = goma_ctx->fem.fv->d_grad_d_dmesh[2][2][2][0];
      goma_ctx->fem.fv->d_div_d_dmesh[2][0] = goma_ctx->fem.fv->d_grad_d_dmesh[0][0][2][0];
      goma_ctx->fem.fv->d_div_d_dmesh[2][0] = goma_ctx->fem.fv->d_grad_d_dmesh[1][1][2][0];
      goma_ctx->fem.fv->d_div_d_dmesh[0][0] = goma_ctx->fem.fv->d_grad_d_dmesh[2][2][0][0];
      goma_ctx->fem.fv->d_div_d_dmesh[1][0] = goma_ctx->fem.fv->d_grad_d_dmesh[2][2][1][0];
    }

    for (j = 1; j < mdofs; j++) {

      /* fv->d_div_d_dmesh[b][j] += fv->d_grad_d_dmesh[p][p] [b][j]; */

      goma_ctx->fem.fv->d_div_d_dmesh[0][j] += goma_ctx->fem.fv->d_grad_d_dmesh[0][0][0][j];
      goma_ctx->fem.fv->d_div_d_dmesh[1][j] += goma_ctx->fem.fv->d_grad_d_dmesh[1][1][1][j];
      goma_ctx->fem.fv->d_div_d_dmesh[0][j] += goma_ctx->fem.fv->d_grad_d_dmesh[1][1][0][j];
      goma_ctx->fem.fv->d_div_d_dmesh[1][j] += goma_ctx->fem.fv->d_grad_d_dmesh[0][0][1][j];

      if (VIMis3) {
        goma_ctx->fem.fv->d_div_d_dmesh[2][j] += goma_ctx->fem.fv->d_grad_d_dmesh[2][2][2][j];
        goma_ctx->fem.fv->d_div_d_dmesh[2][j] += goma_ctx->fem.fv->d_grad_d_dmesh[0][0][2][j];
        goma_ctx->fem.fv->d_div_d_dmesh[2][j] += goma_ctx->fem.fv->d_grad_d_dmesh[1][1][2][j];
        goma_ctx->fem.fv->d_div_d_dmesh[0][j] += goma_ctx->fem.fv->d_grad_d_dmesh[2][2][0][j];
        goma_ctx->fem.fv->d_div_d_dmesh[1][j] += goma_ctx->fem.fv->d_grad_d_dmesh[2][2][1][j];
      }
    }

#endif

  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][MESH_DISPLACEMENT1] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_d_dmesh, 0, siz);
    memset(goma_ctx->fem.fv->d_grad_d_dot_dmesh, 0, siz);
    siz = sizeof(double) * DIM * MDE;
    memset(goma_ctx->fem.fv->d_div_d_dmesh, 0, siz);
  }

  /*
   * d( grad(d_rs) )/dmesh and d( grad(d_rs) ) /dd_rs
   */

  /* Yes, I know it should be each "+p", */
  /* but this way we can collect three */
  /* components of vectors and tensors for */
  /* less than 3 independent spatial dimensions*/

  if (goma_ctx->fem.pd->gv[SOLID_DISPLACEMENT1]) {
    v = SOLID_DISPLACEMENT1;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    siz = sizeof(double) * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_d_rs_dmesh, 0, siz);
    siz = sizeof(double) * DIM * MDE;
    memset(goma_ctx->fem.fv->d_div_d_rs_dmesh, 0, siz);

    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            /*  fv->d_grad_d_rs_dmesh[p][q] [b][j] = 0.; */
            for (r = 0; r < dim; r++) {
              for (i = 0; i < vdofs; i++) {
                goma_ctx->fem.fv->d_grad_d_rs_dmesh[p][q][b][j] +=
                    *goma_ctx->fem.esp->d_rs[r][i] * bfv->d_grad_phi_e_dmesh[i][r][p][q][b][j];
              }
            }
          }
        }
      }
    }

    /*
     * d( div(d_rs) )/dmesh [ just trace(grad(d_rs)) ]
     */
    for (b = 0; b < VIM; b++) {
      for (j = 0; j < mdofs; j++) {
        for (p = 0; p < VIM; p++) {
          goma_ctx->fem.fv->d_div_d_rs_dmesh[b][j] +=
              goma_ctx->fem.fv->d_grad_d_rs_dmesh[p][p][b][j];
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SOLID_DISPLACEMENT1] != -1 &&
             okToZero) {
    siz = sizeof(double) * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_d_rs_dmesh, 0, siz);
    siz = sizeof(double) * DIM * MDE;
    memset(goma_ctx->fem.fv->d_div_d_rs_dmesh, 0, siz);
  }

  /*
   * d(grad(S))/dmesh
   */

  if (goma_ctx->fem.pd->gv[POLYMER_STRESS11]) {
    v = POLYMER_STRESS11;
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    bfv = goma_ctx->fem.bf[v];
    siz = sizeof(double) * MAX_MODES * DIM * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_S_dmesh, 0, siz);
    siz = sizeof(double) * MAX_MODES * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_div_S_dmesh, 0, siz);

    for (mode = 0; mode < goma_ctx->fem.vn->modes; mode++) {
      for (p = 0; p < VIM; p++) {
        for (q = 0; q < VIM; q++) {
          for (r = 0; r < VIM; r++) {
            for (b = 0; b < VIM; b++) {
              for (j = 0; j < mdofs; j++) {
                for (i = 0; i < vdofs; i++) {
                  if (p <= q) {
                    goma_ctx->fem.fv->d_grad_S_dmesh[mode][r][p][q][b][j] +=
                        *goma_ctx->fem.esp->S[mode][p][q][i] * bfv->d_grad_phi_dmesh[i][r][b][j];
                  } else {
                    goma_ctx->fem.fv->d_grad_S_dmesh[mode][r][p][q][b][j] +=
                        *goma_ctx->fem.esp->S[mode][q][p][i] * bfv->d_grad_phi_dmesh[i][r][b][j];
                  }
                }
              }
            }
          }
        }
      }

      for (r = 0; r < VIM; r++) {
        for (q = 0; q < VIM; q++) {
          for (b = 0; b < VIM; b++) {
            for (j = 0; j < mdofs; j++) {
              goma_ctx->fem.fv->d_div_S_dmesh[mode][r][b][j] +=
                  goma_ctx->fem.fv->d_grad_S_dmesh[mode][q][q][r][b][j];
            }
          }
        }
      }

      if (goma_ctx->fem.pd->CoordinateSystem != CARTESIAN) {
        for (s = 0; s < VIM; s++) {
          for (r = 0; r < VIM; r++) {
            for (p = 0; p < VIM; p++) {
              for (q = 0; q < VIM; q++) {
                for (b = 0; b < VIM; b++) {
                  for (j = 0; j < mdofs; j++) {
                    goma_ctx->fem.fv->d_div_S_dmesh[mode][s][b][j] +=
                        goma_ctx->fem.fv->S[mode][p][q] *
                        (goma_ctx->fem.fv->d_grad_e_dq[p][r][q][b] * bfm->phi[j] *
                             (double)delta(s, q) +
                         goma_ctx->fem.fv->d_grad_e_dq[q][p][s][b] * bfm->phi[j] *
                             (double)delta(r, p));
                  }
                }
              }
            }
          }
        }
      }
    } /* end of modal loop */
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][POLYMER_STRESS11] != -1 && okToZero) {
    siz = sizeof(double) * MAX_MODES * DIM * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_S_dmesh, 0, siz);
    siz = sizeof(double) * MAX_MODES * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_div_S_dmesh, 0, siz);
  }

  /*
   * d(grad(G))/dmesh
   */

  if (goma_ctx->fem.pd->gv[VELOCITY_GRADIENT11]) {
    v = VELOCITY_GRADIENT11;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

    siz = sizeof(double) * DIM * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_G_dmesh, 0, siz);
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_div_G_dmesh, 0, siz);
    memset(goma_ctx->fem.fv->d_div_Gt_dmesh, 0, siz);

    for (p = 0; p < VIM; p++) {
      for (q = 0; q < VIM; q++) {
        for (r = 0; r < VIM; r++) {
          for (b = 0; b < VIM; b++) {
            for (j = 0; j < mdofs; j++) {
              for (i = 0; i < vdofs; i++) {
                goma_ctx->fem.fv->d_grad_G_dmesh[r][p][q][b][j] +=
                    *goma_ctx->fem.esp->G[p][q][i] * bfv->d_grad_phi_dmesh[i][r][b][j];
              }
            }
          }
        }
      }
    }

    for (r = 0; r < VIM; r++) {
      for (q = 0; q < VIM; q++) {
        for (b = 0; b < VIM; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_div_G_dmesh[r][b][j] +=
                goma_ctx->fem.fv->d_grad_G_dmesh[q][q][r][b][j];
          }
        }
      }
    }

    if (goma_ctx->fem.pd->CoordinateSystem != CARTESIAN) {
      for (s = 0; s < VIM; s++) {
        for (r = 0; r < VIM; r++) {
          for (p = 0; p < VIM; p++) {
            for (q = 0; q < VIM; q++) {
              for (b = 0; b < dim; b++) {
                for (j = 0; j < mdofs; j++) {
                  goma_ctx->fem.fv->d_div_G_dmesh[s][b][j] +=
                      goma_ctx->fem.fv->G[p][q] * (goma_ctx->fem.fv->d_grad_e_dq[p][r][q][b] *
                                                       bfm->phi[j] * (double)delta(s, q) +
                                                   goma_ctx->fem.fv->d_grad_e_dq[q][p][s][b] *
                                                       bfm->phi[j] * (double)delta(r, p));
                }
              }
            }
          }
        }
      }
    }

    for (r = 0; r < VIM; r++) {
      for (q = 0; q < VIM; q++) {
        for (b = 0; b < VIM; b++) {
          for (j = 0; j < mdofs; j++) {
            goma_ctx->fem.fv->d_div_Gt_dmesh[r][b][j] +=
                goma_ctx->fem.fv->d_grad_G_dmesh[q][r][q][b][j];
          }
        }
      }
    }

    if (goma_ctx->fem.pd->CoordinateSystem != CARTESIAN) {
      for (s = 0; s < VIM; s++) {
        for (r = 0; r < VIM; r++) {
          for (p = 0; p < VIM; p++) {
            for (q = 0; q < VIM; q++) {
              for (b = 0; b < VIM; b++) {
                for (j = 0; j < mdofs; j++) {
                  goma_ctx->fem.fv->d_div_G_dmesh[s][b][j] +=
                      goma_ctx->fem.fv->G[q][p] * (goma_ctx->fem.fv->d_grad_e_dq[p][r][q][b] *
                                                       bfm->phi[j] * (double)delta(s, q) +
                                                   goma_ctx->fem.fv->d_grad_e_dq[q][p][s][b] *
                                                       bfm->phi[j] * (double)delta(r, p));
                }
              }
            }
          }
        }
      }
    }

  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][VELOCITY_GRADIENT11] != -1 &&
             okToZero) {
    siz = sizeof(double) * DIM * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_G_dmesh, 0, siz);
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_div_G_dmesh, 0, siz);
    memset(goma_ctx->fem.fv->d_div_Gt_dmesh, 0, siz);
  }

  /*
   * d(grad(vd))/dmesh
   */
  if (goma_ctx->fem.pd->gv[VORT_DIR1]) {
    v = VORT_DIR1;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

    siz = sizeof(double) * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_vd_dmesh, 0, siz);

    for (r = 0; r < DIM; r++) {
      for (i = 0; i < vdofs; i++) {
        v_ri = *goma_ctx->fem.esp->vd[r][i];
        for (p = 0; p < DIM; p++) {
          for (q = 0; q < DIM; q++) {
            for (b = 0; b < VIM; b++) {
              for (j = 0; j < mdofs; j++) {
                goma_ctx->fem.fv->d_grad_vd_dmesh[p][q][b][j] +=
                    v_ri * bfv->d_grad_phi_e_dmesh[i][r][p][q][b][j];
              }
            }
          }
        }
      }
    }

    /*
     * d( div(vd) )/dmesh
     */

    for (b = 0; b < VIM; b++) {
      for (j = 0; j < mdofs; j++) {
        for (p = 0; p < DIM; p++) {
          goma_ctx->fem.fv->d_div_vd_dmesh[b][j] += goma_ctx->fem.fv->d_grad_vd_dmesh[p][p][b][j];
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][VORT_DIR1] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_vd_dmesh, 0, siz);
  }

  /*
   *   d(n_dot_curl_s_v)/dmesh
   *        This is carried out on a shell
   */
  if (goma_ctx->fem.pd->gv[N_DOT_CURL_V]) {
    v = N_DOT_CURL_V;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_n_dot_curl_s_v_dmesh, 0, siz);
    for (p = 0; p < VIM; p++) {
      for (b = 0; b < dim; b++) {
        for (j = 0; j < mdofs; j++) {
          for (i = 0; i < vdofs; i++) {
            goma_ctx->fem.fv->d_grad_n_dot_curl_s_v_dmesh[p][b][j] +=
                (*goma_ctx->fem.esp->n_dot_curl_s_v[i] * bfv->d_grad_phi_dmesh[i][p][b][j]);
          }
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][N_DOT_CURL_V] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_n_dot_curl_s_v_dmesh, 0, siz);
  }

  /*
   *   d(grad_div_s_v_dmesh[b][jvar][jShell]);
   *        This is carried out on a shell
   */
  if (goma_ctx->fem.pd->gv[SHELL_SURF_DIV_V]) {
    v = SHELL_SURF_DIV_V;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_div_s_v_dmesh, 0, siz);
    for (p = 0; p < VIM; p++) {
      for (b = 0; b < dim; b++) {
        for (j = 0; j < mdofs; j++) {
          for (i = 0; i < vdofs; i++) {
            goma_ctx->fem.fv->d_grad_div_s_v_dmesh[p][b][j] +=
                (*goma_ctx->fem.esp->div_s_v[i] * bfv->d_grad_phi_dmesh[i][p][b][j]);
          }
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_SURF_DIV_V] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_div_s_v_dmesh, 0, siz);
  }

  /*
   *   d_grad_curv_dmesh[b][jvar][jShell])
   *        This is carried out on a shell
   */
  if (goma_ctx->fem.pd->gv[SHELL_SURF_CURV]) {
    v = SHELL_SURF_CURV;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_curv_dmesh, 0, siz);
    for (p = 0; p < VIM; p++) {
      for (b = 0; b < dim; b++) {
        for (j = 0; j < mdofs; j++) {
          for (i = 0; i < vdofs; i++) {
            goma_ctx->fem.fv->d_grad_curv_dmesh[p][b][j] +=
                (*goma_ctx->fem.esp->curv[i] * bfv->d_grad_phi_dmesh[i][p][b][j]);
          }
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_SURF_CURV] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_grad_curv_dmesh, 0, siz);
  }

  /*
   *   d_serialgrad_div_s_v_dmesh[b][jvar][jShell]);
   *        This is carried out on a shell
   */
  if (goma_ctx->fem.pd->gv[GRAD_S_V_DOT_N1]) {
    for (r = 0; r < dim; r++) {
      v = GRAD_S_V_DOT_N1 + r;
      bfv = goma_ctx->fem.bf[v];
      vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];
      siz = sizeof(double) * DIM * DIM * DIM * MDE;
      memset(goma_ctx->fem.fv->d_serialgrad_grad_s_v_dot_n_dmesh, 0, siz);
      for (p = 0; p < VIM; p++) {
        for (b = 0; b < dim; b++) {
          for (j = 0; j < mdofs; j++) {
            for (i = 0; i < vdofs; i++) {
              goma_ctx->fem.fv->d_serialgrad_grad_s_v_dot_n_dmesh[r][p][b][j] +=
                  (*goma_ctx->fem.esp->grad_v_dot_n[r][i] * bfv->d_grad_phi_dmesh[i][p][b][j]);
            }
          }
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][GRAD_S_V_DOT_N1] != -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * DIM * MDE;
    memset(goma_ctx->fem.fv->d_serialgrad_grad_s_v_dot_n_dmesh, 0, siz);
  }

  /*
   * d(grad(sh_J))/dmesh
   */
  if (goma_ctx->fem.pd->gv[SHELL_DIFF_FLUX]) {
    v = SHELL_DIFF_FLUX;
    bfv = goma_ctx->fem.bf[v];
    vdofs = goma_ctx->fem.ei[goma_ctx->fem.upd->matrix_index[v]]->dof[v];

    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_J_dmesh[0][0][0]), 0, siz);

    for (p = 0; p < VIM; p++) {
      for (b = 0; b < dim; b++) {
        for (j = 0; j < mdofs; j++) {
          for (i = 0; i < vdofs; i++) {
            goma_ctx->fem.fv->d_grad_sh_J_dmesh[p][b][j] +=
                *goma_ctx->fem.esp->sh_J[i] * bfv->d_grad_phi_dmesh[i][p][b][j];
          }
        }
      }
    }
  } else if (goma_ctx->fem.upd->vp[goma_ctx->fem.pg->imtrx][SHELL_DIFF_FLUX] == -1 && okToZero) {
    siz = sizeof(double) * DIM * DIM * MDE;
    memset(&(goma_ctx->fem.fv->d_grad_sh_J_dmesh[0][0][0]), 0, siz);
  }

  /*
   * Now load up the sensitivity of the volume element to the
   * mesh variables...
   *
   * d ( h3 ) / d ( mesh_bj )
   */
  if (goma_ctx->fem.pd->gv[MESH_DISPLACEMENT1] == 1) {
    for (b = 0; b < VIM; b++) {
      for (j = 0; j < mdofs; j++) {
        goma_ctx->fem.fv->dh3dmesh[b][j] = goma_ctx->fem.fv->dh3dq[b] * bfm->phi[j];
      }
    }
  }

  return (status);
}
