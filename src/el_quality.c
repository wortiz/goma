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
 *$Id: el_quality.c,v 5.5 2010-03-17 22:23:53 hkmoffa Exp $
 */

#include <math.h>
#include <stdio.h>

#ifdef GOMA_ENABLE_AZTEC
#include "az_aztec.h"
#endif
#include "dp_utils.h"
#include "el_elm.h"
#include "el_elm_info.h"
#include "el_geom.h"
#include "exo_struct.h"
#include "mm_as.h"
#include "mm_as_structs.h"
#include "mm_eh.h"
#include "mm_fill_ptrs.h"
#include "mm_fill_util.h"
#include "mm_unknown_map.h"
#include "rd_mesh.h"
#include "rf_allo.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_mp.h"
#include "std.h"

#define GOMA_EL_QUALITY_C
#include "el_quality.h"

/*************** R O U T I N E S   I N   T H I S   F I L E ********************
 *
 *  NAME				TYPE		CALL_BY
 * ---------------		-------		------------------------
 *  element_quality ()           int             solve_problem
 *  jacobian_metric ()           double          element_quality
 *  angle_metric ()              double          element_quality
 *  triangle_metric ()           double          element_quality
 *  load_vertex_xy ()            void            angle_metric, triangle_metric
 *  vertex_angle ()              double          angle_metric
 *  sidelength ()                double          triangle_metric
 *
 ******************************************************************************/

/* Local function prototypes */
static double jacobian_metric(Exo_DB *, /* Exodus database structure */
                              double *, /* Solution vector */
                              int *);   /* proc_config array */
static double volume_metric(int *);     /* proc_config array */
static double angle_metric(Exo_DB *,    /* Exodus database structure */
                           double *,    /* Solution vector */
                           int *);      /* proc_config array */
static double triangle_metric(Exo_DB *, /* Exodus database structure */
                              double *, /* Solution vector */
                              int *);   /* proc_config array */
static void load_vertex_xy(Exo_DB *,    /* Exodus database structure */
                           int,         /* Element number */
                           int,         /* Number of nodes to process */
                           double *,    /* Solution vector */
                           double **);  /* Vertex coordinates */
static double vertex_angle(double **,   /* Vertex coordinates */
                           int,         /* Current vertex */
                           int,         /* Number of perimeter nodes (4 or 8) */
                           int *);      /* Node numbering sense (CW=+1, CCW=-1) */
static double sidelength(int,           /* One vertex */
                         int,           /* Other vertex */
                         double **);    /* Vertex coordinates */

int element_quality(Exo_DB *exo, double *x, int *proc_config)

/*
 *   Function which computes measures of element quality
 *   (lack of distortion) and compares to specified criterion
 *   to determine when a transient run should be stopped to
 *   do remeshing/remapping prior to continuing.
 *
 *      Author:          Edward D. Wilkes (9233)
 *      Date:            16 Oct 2002
 *      Revised:
 *
 *   The routine currently handles the following metrics of
 *   quadrilateral finite element quality/distortion:
 *
 *	jac   		Minimum element Jacobian at Gauss points
 *			(from Bach & Hassager, JFM 1985)
 *	ang		Maximum deviation of interior angles from 90 degrees
 *			(from El-Hamalawi, Comp & Struct 2000)
 *      tri		Maximum distortion of sub-triangles
 *			(from El-Hamalawi, Comp & Struct 2000)
 *
 */

{
  double mavg = 0.0, tavg = 0.0, quality = 0.0;
  double qmin = 999.9, wt_sum = 0.0, wt_min = 0.0;
  static int first_call = TRUE;

  /* Quick exit if no metrics were specified */
  if (nEQM == 0)
    return (TRUE);

  /* Output table header */
  DPRINTF(stderr, "\n ELEMENT QUALITY METRIC         AVG              MIN\n");

  /* Compute each requested metric */
  if (goma_ctx->fem.eqm->do_jac) {
    mavg = jacobian_metric(exo, x, proc_config);
    DPRINTF(stderr, "               Jacobian         %8g         %8g\n", mavg,
            goma_ctx->fem.eqm->eq_jac);
    tavg += goma_ctx->fem.eqm->wt_jac * mavg;
    wt_min += goma_ctx->fem.eqm->wt_jac * goma_ctx->fem.eqm->eq_jac;
    wt_sum += goma_ctx->fem.eqm->wt_jac;
    if (goma_ctx->fem.eqm->eq_jac < qmin)
      qmin = goma_ctx->fem.eqm->eq_jac;
  }
  if (goma_ctx->fem.eqm->do_vol && !first_call) {
    mavg = volume_metric(proc_config);
    DPRINTF(stderr, "               Volume change    %8g         %8g\n", mavg,
            goma_ctx->fem.eqm->eq_vol);
    tavg += goma_ctx->fem.eqm->wt_vol * mavg;
    wt_min += goma_ctx->fem.eqm->wt_vol * goma_ctx->fem.eqm->eq_vol;
    wt_sum += goma_ctx->fem.eqm->wt_vol;
    if (goma_ctx->fem.eqm->eq_vol < qmin)
      qmin = goma_ctx->fem.eqm->eq_vol;
  }
  if (goma_ctx->fem.eqm->do_ang) {
    mavg = angle_metric(exo, x, proc_config);
    DPRINTF(stderr, "               Angle            %8g         %8g\n", mavg,
            goma_ctx->fem.eqm->eq_ang);
    tavg += goma_ctx->fem.eqm->wt_ang * mavg;
    wt_min += goma_ctx->fem.eqm->wt_ang * goma_ctx->fem.eqm->eq_ang;
    wt_sum += goma_ctx->fem.eqm->wt_ang;
    if (goma_ctx->fem.eqm->eq_ang < qmin)
      qmin = goma_ctx->fem.eqm->eq_ang;
  }
  if (goma_ctx->fem.eqm->do_tri) {
    mavg = triangle_metric(exo, x, proc_config);
    DPRINTF(stderr, "               Triangle         %8g         %8g\n", mavg,
            goma_ctx->fem.eqm->eq_tri);
    tavg += goma_ctx->fem.eqm->wt_tri * mavg;
    wt_min += goma_ctx->fem.eqm->wt_tri * goma_ctx->fem.eqm->eq_tri;
    wt_sum += goma_ctx->fem.eqm->wt_tri;
    if (goma_ctx->fem.eqm->eq_tri < qmin)
      qmin = goma_ctx->fem.eqm->eq_tri;
  }

  /* Combined metric based on each requested method */
  tavg /= wt_sum;
  wt_min /= wt_sum;
  if (nEQM > 1) {
    DPRINTF(stderr, "               COMBINED         %8g         %8g     %8g\n", tavg, qmin,
            wt_min);
  }

  /* Assign quality value according to specified tolerance type */
  if (goma_ctx->fem.eqm->tol_type == EQM_AVG) {
    quality = tavg;
  } else if (goma_ctx->fem.eqm->tol_type == EQM_JAC && goma_ctx->fem.eqm->do_jac) {
    quality = goma_ctx->fem.eqm->eq_jac;
  } else if (goma_ctx->fem.eqm->tol_type == EQM_VOL && goma_ctx->fem.eqm->do_vol) {
    quality = goma_ctx->fem.eqm->eq_vol;
  } else if (goma_ctx->fem.eqm->tol_type == EQM_ANG && goma_ctx->fem.eqm->do_ang) {
    quality = goma_ctx->fem.eqm->eq_ang;
  } else if (goma_ctx->fem.eqm->tol_type == EQM_TRI && goma_ctx->fem.eqm->do_tri) {
    quality = goma_ctx->fem.eqm->eq_tri;
  } else if (goma_ctx->fem.eqm->tol_type == EQM_WTMIN) {
    quality = wt_min;
  } else {
    quality = qmin;
  }

  /* Check quality against tolerance and return */
  first_call = FALSE;
  if (quality < goma_ctx->fem.eqm->eq_tol) {
    DPRINTF(stderr, "Element quality below tolerance of %g\n", goma_ctx->fem.eqm->eq_tol);
    DPRINTF(stderr, "\tREMESHING IS REQUIRED!\n");
    return (FALSE);
  } else {
    DPRINTF(stderr, "Element quality OK!\n");
    return (TRUE);
  }
} /* End of function "element_quality" */

static double jacobian_metric(Exo_DB *exo, double *x, int *proc_config) {
  int dofs, k;
  int ielem, e_start, e_end, igp, ngp, store_shape;
  double gwt, Jw, Jw_sum, Jw_min, els, eq, eqavg;
  double eqsum = 0.0, eqmin = 999.9;
  double dj00, dj01, dj10, dj11, detJ;
  double xi[DIM];
  double **xy = NULL;
  /*  struct Basis_Functions *bd;  */
  BASIS_FUNCTIONS_STRUCT *bd;

  /* Allocate vertex coordinate array */
  xy = (double **)array_alloc(2, 2, 9, sizeof(double));

  /* Loop over elements */
  e_start = exo->eb_ptr[0];
  e_end = exo->eb_ptr[exo->num_elem_blocks];
  for (ielem = e_start; ielem < e_end; ielem++) {

    /* Set up ei pointers and get node coordinates */
    bd = ((goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][R_MESH1])
              ? goma_ctx->fem.bf[R_MESH1]
              : goma_ctx->fem.bf[goma_ctx->fem.pd->ShapeVar]);
    load_ei(ielem, exo, 0, goma_ctx->fem.pg->imtrx);
    ngp = elem_info(NQUAD, goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->ielem_type);
    dofs = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[goma_ctx->fem.pd->ShapeVar];
    load_vertex_xy(exo, ielem, dofs, x, xy);
    store_shape = bd->element_shape;

    /* Loop over Gauss quadrature points */
    if (goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->ielem_dim == 2) {
      if (bd->element_shape != goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->ielem_shape) {
        bd->element_shape = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->ielem_shape;
      }
      Jw_sum = 0.0;
      Jw_min = 99999.9;
      for (igp = 0; igp < ngp; igp++) {

        /* Find elemental Jacobian determinant and Gauss weight for each point */
        dj00 = 0.0;
        dj01 = 0.0;
        dj10 = 0.0;
        dj11 = 0.0;
        find_stu(igp, goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->ielem_type, &xi[0], &xi[1],
                 &xi[2]);
        load_basis_functions(xi, goma_ctx->fem.bfd);
        gwt = Gq_weight(igp, goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->ielem_type);

        /* Sum components of elemental Jacobian */
        for (k = 0; k < dofs; k++) {
          dj00 += bd->dphidxi[k][0] * xy[0][k];
          dj01 += bd->dphidxi[k][0] * xy[1][k];
          dj10 += bd->dphidxi[k][1] * xy[0][k];
          dj11 += bd->dphidxi[k][1] * xy[1][k];
        }

        /* Evaluate determinant detJ */
        detJ = dj00 * dj11 - dj01 * dj10;
        /*          Jw = detJ * gwt;	*/
        Jw = fabs(detJ) * gwt;
        Jw_sum += Jw;
        if (Jw < Jw_min)
          Jw_min = Jw;
      }

      /* Calculate distortion for current element */
      eq = (double)ngp * Jw_min / Jw_sum;
      eqsum += eq;
      if (eq < eqmin)
        eqmin = eq;
    } else {
      GOMA_WH(-1, "non 2D elements in Jacobian Quality\n");
    }
    /*    restore element shape to its original value  */
    bd->element_shape = store_shape;
  }

  /* Return results */
  els = (double)(e_end - e_start);
  if (goma_ctx->num_proc > 1) {
    els = goma_gsum_double(els);
    eqsum = goma_gsum_double(eqsum);
    eqmin = goma_gmin_double(eqmin);
  }
  eqavg = eqsum / els;
  goma_ctx->fem.eqm->eq_jac = eqmin;
  return eqavg;
} /* End of function jacobian_metric */

static double volume_metric(int *proc_config)
/*
 * Volume change data, if requested, were collected during mesh assembly.
 * Now, just do parallel processing and report back.
 */
{
  double points = (int)goma_ctx->fem.eqm->vol_count;
  double sum = goma_ctx->fem.eqm->vol_sum;
  double low = goma_ctx->fem.eqm->vol_low;

  if (goma_ctx->num_proc > 1) {
    points = (int)goma_gsum_int(goma_ctx->fem.eqm->vol_count);
    sum = goma_gsum_double(goma_ctx->fem.eqm->vol_sum);
    low = goma_gmin_double(goma_ctx->fem.eqm->vol_low);
  }

  goma_ctx->fem.eqm->eq_vol = low;

  if (points == 0.0) {
    return -1.0;
  } else {
    return sum / points;
  }
} /* End of function volume_metric */

static double angle_metric(Exo_DB *exo, double *x, int *proc_config) {
  int i, ielem, e_start, e_end, sense;
  /* int e_sens */
  int bad_elem = FALSE;
  int n, nn;
  double angle, delta, delta_sum, f, els;
  double eq, eqavg, eqsum = 0.0, eqmin = 9999.9;
  double **xy = NULL;

  /* Allocate vertex coordinate array */
  xy = (double **)array_alloc(2, 2, 8, sizeof(double));

  /* Loop over elements */
  e_start = exo->eb_ptr[0];
  e_end = exo->eb_ptr[exo->num_elem_blocks];
  for (ielem = e_start; ielem < e_end; ielem++) {

    /* Determine number of perimeter nodes */
    nn = elem_info(NNODES, Elem_Type(exo, ielem));
    if (nn == 8 || nn == 9) {
      n = 8;
    } else {
      n = 4;
    }

    /* Get vertex coordinates */
    load_vertex_xy(exo, ielem, n, x, xy);

    /* Loop over local element vertices */
    for (i = 0; i < 4; i++) {

      /* Get vertex angle */
      delta_sum = 0.0;
      angle = vertex_angle(xy, i, n, &sense);

      /* First local vertex: set numbering sense */
      if (i == 0) {
        /* e_sens = sense; */
      }

      /* Other vertices: check against reference sense */
      /*
        else if (sense != e_sens)
        {
        DPRINTF(stderr, "P%d:  Element %d appears to be concave!",
        ProcID, ielem);
        bad_elem = TRUE;
        }

      */
      /* Calculate and sum angle deviation from 90 degrees */
      delta = fabs(angle - 0.5 * M_PIE);
      delta_sum += delta * delta;
    }

    /* Calculate L2 norm of angle deviation and element quality, */
    /* scale such that 1/2 indicates a deviation norm of 45 degrees */
    f = sqrt(delta_sum / 4.0);
    eq = 1.0 - f / (0.5 * M_PIE);
    eqsum += eq;
    if (eq < eqmin)
      eqmin = eq;
  }

  /* Set quality to -1 if a bad element (angle over 180 degrees) was found */
  if (bad_elem)
    goma_ctx->fem.eqm->eq_ang = -1.0;

  /* Return results */
  els = (double)(e_end - e_start);
  if (goma_ctx->num_proc > 1) {
    els = goma_gsum_double(els);
    eqsum = goma_gsum_double(eqsum);
    eqmin = goma_gmin_double(eqmin);
  }
  eqavg = eqsum / els;
  if (!bad_elem)
    goma_ctx->fem.eqm->eq_ang = eqmin;

  safer_free((void **)&xy);
  return eqavg;
} /* End of function angle_metric */

static double triangle_metric(Exo_DB *exo, double *x, int *proc_config) {
  int i, ielem, e_start, e_end, sense;
  /* e_sens */
  int v1, v2, v3;
  int bad_elem = FALSE;
  int isort[4], jsort[4];
  double alpha[4];
  double s12, s13, s23;
  double angle, els;
  double eq, eqavg, eqsum = 0.0, eqmin = 9999.9;
  double **xy = NULL;

  /* Allocate vertex coordinate array */
  xy = (double **)array_alloc(2, 2, 4, sizeof(double));

  /* Loop over elements */
  e_start = exo->eb_ptr[0];
  e_end = exo->eb_ptr[exo->num_elem_blocks];
  for (ielem = e_start; ielem < e_end; ielem++) {

    /* Get vertex coordinates */
    load_vertex_xy(exo, ielem, 4, x, xy);

    /* Analyze Lo's alpha factor for each of four subtriangles */
    for (i = 0; i < 4; i++) {
      alpha[i] = 0.0;
      isort[i] = 0;
      jsort[i] = 0;
    }
    for (v2 = 0; v2 < 4; v2++) {
      v1 = ((v2 == 0) ? 3 : v2 - 1);
      v3 = ((v2 == 3) ? 0 : v2 + 1);
      s12 = sidelength(v1, v2, xy);
      s13 = sidelength(v1, v3, xy);
      s23 = sidelength(v2, v3, xy);
      angle = vertex_angle(xy, v2, 4, &sense);
      alpha[v2] = (s12 * s23 * sin(angle)) / (s12 * s12 + s13 * s13 + s23 * s23);

      /* For first subtriangle, record node numbering sense */
      if (v2 == 0) {
        /* e_sens = sense; */
      }

      /* For others, check sense and update rank arrays */
      else {

        /* Detect concave angle, mark as bad if found */
        /*
          if (sense != e_sens)
          {
          DPRINTF(stderr, "P%d:  Element %d appears to be concave!",
          ProcID, ielem);
          bad_elem = TRUE;
          }
        */

        for (i = 0; i < v2; i++) {
          if (alpha[v2] > alpha[i]) {
            jsort[v2]++;
          } else {
            jsort[i]++;
          }
        }
      }
    }

    /* Set rank index */
    for (i = 0; i < 4; i++) {
      isort[jsort[i]] = i;
    }

    /* Calculate quadrilateral factor (two lowest / two highest), check for zero */

    if (bad_elem || alpha[isort[3]] == 0.0) {
      eq = 0.0;
    } else

    {
      eq = (alpha[isort[0]] * alpha[isort[1]]) / (alpha[isort[2]] * alpha[isort[3]]);
    }

    eqsum += eq;
    if (eq < eqmin)
      eqmin = eq;
  }

  /* Set quality to -1 if a bad element (angle over 180 degrees) was found */
  if (bad_elem)
    goma_ctx->fem.eqm->eq_tri = -1.0;

  /* Return results */
  els = (double)(e_end - e_start);
  if (goma_ctx->num_proc > 1) {
    els = goma_gsum_double(els);
    eqsum = goma_gsum_double(eqsum);
    eqmin = goma_gmin_double(eqmin);
  }
  eqavg = eqsum / els;
  if (!bad_elem)
    goma_ctx->fem.eqm->eq_tri = eqmin;

  safer_free((void **)&xy);
  return eqavg;
} /* End of function triangle_metric */

static void load_vertex_xy(Exo_DB *exo, int ielem, int dofs, double *x, double **xy) {
  int node, i, k, index, nd;
  int DM = FALSE;

  load_ei(ielem, exo, 0, goma_ctx->fem.pg->imtrx);
  DM = (goma_ctx->fem.pd_glob[goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->mn]
            ->e[goma_ctx->fem.pg->imtrx][R_MESH1]);
  for (i = 0; i < goma_ctx->fem.pd->Num_Dim; i++) {
    for (k = 0; k < dofs; k++) {
      if (DM) {
        node = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof_list[R_MESH1 + i][k];
        index = goma_ctx->mesh.proc_elem_connect[goma_ctx->mesh.proc_connect_ptr[ielem] + node];
        nd = Index_Solution(index, MESH_DISPLACEMENT1 + i, 0, 0,
                            goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->mn, goma_ctx->fem.pg->imtrx);
        GOMA_EH(nd, "Bad displacement unknown index!");
        xy[i][k] = goma_ctx->mesh.coor[i][index] + x[nd];
      } else {
        node = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof_list[goma_ctx->fem.pd->ShapeVar][k];
        index = goma_ctx->mesh.proc_elem_connect[goma_ctx->mesh.proc_connect_ptr[ielem] + node];
        xy[i][k] = goma_ctx->mesh.coor[i][index];
      }
    }
  }
  return;
}

static double vertex_angle(double **xy, int i, int n, int *sense) {
  int v1 = -100, v2, v3 = -100;
  double mx, my, mJ, x1, x2, x3, y1, y2, y3, rx3, ry3, l23;

  /* Vertex v2 angle is calculated on this call */
  v2 = i;

  /* For 4 node elements, all are on perimeter */
  if (n == 4) {
    v1 = ((v2 == 0) ? 3 : i - 1);
    v3 = ((v2 == 3) ? 0 : i + 1);
  }

  /* For 8 or 9 node elements, use mid-side nodes too */
  else if (n == 8) {
    v3 = v2 + 4;
    v1 = ((v2 == 0) ? 7 : v3 - 1);
  } else {
    GOMA_EH(GOMA_ERROR, "vertex_angle: n must be 4 or 8, or else there is an algorithm error");
  }

  /* Load coordinates of v2 and the two neighbor nodes */
  x1 = xy[0][v1];
  x2 = xy[0][v2];
  x3 = xy[0][v3];
  y1 = xy[1][v1];
  y2 = xy[1][v2];
  y3 = xy[1][v3];

  /* Rotate the angle such that v2 is at origin and v1 is on positive x-axis */
  mx = x1 - x2;
  my = y1 - y2;
  mJ = sqrt(mx * mx + my * my);

  /* Now, rotated v1 coordinates are just (mJ, 0) */
  /* Transform to get rotated v3 coordinates */
  rx3 = (mx * (x3 - x2) + my * (y3 - y2)) / mJ;
  ry3 = (-my * (x3 - x2) + mx * (y3 - y2)) / mJ;

  /*
   * If local node numbering is clockwise, ry3 should be positive, and if
   * counterclockwise, ry3 should be negative. Either way, a change in the
   * sign of ry3 indicates an angle larger than pi (or 180 degrees).
   * Pass this sign back via pointer to check.
   */
  *sense = SGN(ry3);

  /* Cosine of v2 angle is rx3 / hypoteneuse (l23) */
  l23 = sqrt(rx3 * rx3 + ry3 * ry3);
  return acos(rx3 / l23);
}

static double sidelength(int v1, int v2, double **xy) {
  return sqrt((xy[0][v1] - xy[0][v2]) * (xy[0][v1] - xy[0][v2]) +
              (xy[1][v1] - xy[1][v2]) * (xy[1][v1] - xy[1][v2]));
}

/* END of file el_quality.c  */
/*****************************************************************************/
