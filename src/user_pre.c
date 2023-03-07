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

/*
 *$Id: user_pre.c,v 5.1 2007-09-18 18:53:49 prschun Exp $
 */

#include <math.h>
#include <stdio.h>

/* GOMA include files */

#include "dpi.h"
#include "el_elm.h"
#include "mm_as.h"
#include "mm_as_structs.h"
#include "mm_eh.h"
#include "mm_unknown_map.h"
#include "rf_fem_const.h"
#include "std.h"
#include "user_pre.h"

#define GOMA_USER_PRE_C

/*********** R O U T I N E S   I N   T H I S   F I L E ************************
 *
 *       NAME            TYPE            CALLED_BY
 *    user_init_object                    object_distance
 *    user_mat_init                       rf_util.c/
 *    ------------             ---------               --------------
 *************************/
double user_surf_object(int *int_params, dbl *param, dbl *r) {
  double distance = 0;
  static int warning = 0;

  /*
  int num_params;

  num_params = int_params[0];
  */

  /**********************************************************/

  /* Comment out our remove this line if using this routine */
  if (warning == 0) {
    DPRINTF(stderr, "\n\n#############\n"
                    "# WARNING!! #  No user_defined surface object model implemented"
                    "\n#############\n");
    warning = 1;
  }

  return distance;
} /* End of routine user_init_object */

double user_mat_init(const int var,
                     const int node,
                     const double init_value,
                     const double p[],
                     const double xpt[],
                     const int mn,
                     const double var_vals[]) {

  double value = 0;
  /* Set this to a nonzero value if using this routine */
  static int warning = -1;

  if (warning == 0) {
    DPRINTF(stderr, "\n\n#############\n"
                    "# WARNING!! #  No user_defined material initialization model implemented"
                    "\n#############\n");
    warning = 1;
  } else if (var == TEMPERATURE) {
    double xpt0[DIM], dist, distz, T_below, T_init;
    double alpha, speed, ht, sum, xn, exp_arg;
    int n_terms = 4, nt, dir;
    for (dir = 0; dir < DIM; dir++) {
      xpt0[dir] = p[dir];
    }
    alpha = p[DIM];
    speed = p[DIM + 1];
    ht = p[DIM + 2];
    T_below = p[DIM + 3];
    T_init = init_value;
    sum = 0.;
    for (nt = 0; nt < n_terms; nt++) {
      xn = 0.5 + ((double)nt);
      dist = 0.;
      for (dir = 0; dir < DIM; dir++) {
        dist += efv->ext_fld_ndl_val[dir][node] * (xpt[dir] - xpt0[dir]);
      }
      exp_arg = dist * alpha * SQUARE(xn * M_PIE / ht) / speed;
      distz = xpt0[2] + 0.5 * ht - xpt[2];
      sum += exp(exp_arg) * cos(M_PIE / ht * distz * xn) * 2. / M_PIE * pow(-1., nt) / xn;
    }
    value = fmin(T_below - (T_below - T_init) * sum, T_init);
    value = fmax(value, T_below);
    if (value < 0) {
      fprintf(stderr, "Trouble, negative temperature! %g %g %g %g\n", value, sum, exp_arg, dist);
    }
  } else if (var >= MESH_DISPLACEMENT1 && var <= MESH_DISPLACEMENT3) {
    double xpt0[DIM], T_pos, T_ref;
    int dir;
    for (dir = 0; dir < DIM; dir++) {
      xpt0[dir] = p[dir];
    }
    T_pos = var_vals[TEMPERATURE];
    T_ref = p[DIM + 1];
    dir = var - MESH_DISPLACEMENT1;
    value = p[DIM] * (xpt[dir] - xpt0[dir]) * (T_pos - T_ref);

  } else {
    GOMA_EH(GOMA_ERROR, "Not a supported usermat initialization condition ");
  }
  return value;
} /* End of routine user_mat_init */

int user_initialize(const int var,
                    double *x,
                    const double init_value,
                    const double p[],
                    const double xpt[],
                    const double var_vals[]) {

  double value = 0;
  int i, var_somewhere, idv, mn;
  /* Set this to a nonzero value if using this routine */
  static int warning = -1;

  if (warning == 0) {
    DPRINTF(stderr, "\n\n#############\n"
                    "# WARNING!! #  No user_defined material initialization model implemented"
                    "\n#############\n");
    warning = 1;
  }

  if (var > -1) {
    if (upd->vp[pg->imtrx][var] > -1) {
      for (i = 0; i < DPI_ptr->num_owned_nodes; i++) {
        var_somewhere = FALSE;
        for (mn = 0; mn < upd->Num_Mat; mn++) {
          idv = Index_Solution(i, var, 0, 0, mn, pg->imtrx);
          if (idv != -1) {
            var_somewhere = TRUE;
            break;
          }
        }
        if (var_somewhere) {
          if (var == TEMPERATURE) {
            double dist, alpha, speed, ht, T_below, T_init, sum, xn, exp_arg;
            int n_terms = 4, nt, dir;
            alpha = p[0];
            speed = p[1];
            ht = p[2];
            T_below = p[3];
            T_init = init_value;
            sum = 0.;
            for (nt = 0; nt < n_terms; nt++) {
              xn = 0.5 + ((double)nt);
              dist = 0.;
              for (dir = 0; dir < DIM; dir++) {
                dist += SQUARE(xpt[dir]);
              }
              dist = sqrt(dist);
              exp_arg = dist * alpha * SQUARE(xn * M_PIE / ht) / speed;
              sum += exp(exp_arg) * cos(M_PIE / ht * dist * xn) * 2. / M_PIE * pow(-1., nt) / xn;
            }
            value = fmin(T_below - (T_below - T_init) * sum, T_init);
            value = fmax(value, T_below);
            if (value < 0) {
              fprintf(stderr, "Trouble, negative temperature! %g %g %g %g\n", value, sum, exp_arg,
                      dist);
            }
            x[idv] = value;
          } else {
            GOMA_EH(GOMA_ERROR, "Not a supported user initialization condition ");
          }
        }
      }
    }
  }

  return 1;
} /* End of routine user_initialize */

/*****************************************************************************/
/* End of file user_pre.c*/
/*****************************************************************************/
