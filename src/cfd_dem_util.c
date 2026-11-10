/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) 2026 Goma Developers, National Technology & Engineering   *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

/*
 * Mesh-independent CFD-DEM helpers.  No Goma globals are touched here so the routines
 * can be unit tested in isolation.  Parametric conventions match rf_shape.c so that an
 * xi found here can be handed straight to Goma's basis function machinery.
 */

#include "cfd_dem_util.h"

#include <math.h>
#include <string.h>

#include "cfd_dem_coupling.h"
#include "el_elm.h"

int cfd_dem_num_vertices(int shape) {
  switch (shape) {
  case TRIANGLE:
    return 3;
  case QUADRILATERAL:
    return 4;
  case TETRAHEDRON:
    return 4;
  case HEXAHEDRON:
    return 8;
  default:
    return -1;
  }
}

int cfd_dem_shape_dim(int shape) {
  switch (shape) {
  case TRIANGLE:
  case QUADRILATERAL:
    return 2;
  case TETRAHEDRON:
  case HEXAHEDRON:
    return 3;
  default:
    return -1;
  }
}

int cfd_dem_vertex_shape(int shape, const double xi[3], double N[], double dN[][3]) {
  const double s = xi[0], t = xi[1], u = xi[2];
  int i;

  switch (shape) {
  case TRIANGLE:
    /* rf_shape.c LINEAR_TRI: N0 = s, N1 = t, N2 = 1 - s - t */
    N[0] = s;
    N[1] = t;
    N[2] = 1.0 - s - t;
    if (dN != NULL) {
      dN[0][0] = 1.0, dN[0][1] = 0.0;
      dN[1][0] = 0.0, dN[1][1] = 1.0;
      dN[2][0] = -1.0, dN[2][1] = -1.0;
      for (i = 0; i < 3; i++)
        dN[i][2] = 0.0;
    }
    return 0;

  case QUADRILATERAL: {
    static const double sv[4] = {-1, 1, 1, -1};
    static const double tv[4] = {-1, -1, 1, 1};
    for (i = 0; i < 4; i++) {
      N[i] = 0.25 * (1.0 + sv[i] * s) * (1.0 + tv[i] * t);
      if (dN != NULL) {
        dN[i][0] = 0.25 * sv[i] * (1.0 + tv[i] * t);
        dN[i][1] = 0.25 * tv[i] * (1.0 + sv[i] * s);
        dN[i][2] = 0.0;
      }
    }
    return 0;
  }

  case TETRAHEDRON:
    /* rf_shape.c LINEAR_TET: N0 = 1 - s - t - u, N1 = s, N2 = t, N3 = u */
    N[0] = 1.0 - s - t - u;
    N[1] = s;
    N[2] = t;
    N[3] = u;
    if (dN != NULL) {
      dN[0][0] = -1.0, dN[0][1] = -1.0, dN[0][2] = -1.0;
      dN[1][0] = 1.0, dN[1][1] = 0.0, dN[1][2] = 0.0;
      dN[2][0] = 0.0, dN[2][1] = 1.0, dN[2][2] = 0.0;
      dN[3][0] = 0.0, dN[3][1] = 0.0, dN[3][2] = 1.0;
    }
    return 0;

  case HEXAHEDRON: {
    static const double sv[8] = {-1, 1, 1, -1, -1, 1, 1, -1};
    static const double tv[8] = {-1, -1, 1, 1, -1, -1, 1, 1};
    static const double uv[8] = {-1, -1, -1, -1, 1, 1, 1, 1};
    for (i = 0; i < 8; i++) {
      const double fs = 1.0 + sv[i] * s, ft = 1.0 + tv[i] * t, fu = 1.0 + uv[i] * u;
      N[i] = 0.125 * fs * ft * fu;
      if (dN != NULL) {
        dN[i][0] = 0.125 * sv[i] * ft * fu;
        dN[i][1] = 0.125 * tv[i] * fs * fu;
        dN[i][2] = 0.125 * uv[i] * fs * ft;
      }
    }
    return 0;
  }

  default:
    return -1;
  }
}

void cfd_dem_shape_center(int shape, double xi[3]) {
  xi[0] = xi[1] = xi[2] = 0.0;
  switch (shape) {
  case TRIANGLE:
    xi[0] = xi[1] = 1.0 / 3.0;
    break;
  case TETRAHEDRON:
    xi[0] = xi[1] = xi[2] = 0.25;
    break;
  default:
    break;
  }
}

int cfd_dem_xi_inside(int shape, const double xi[3], double tol) {
  switch (shape) {
  case TRIANGLE:
    return xi[0] >= -tol && xi[1] >= -tol && (1.0 - xi[0] - xi[1]) >= -tol;
  case QUADRILATERAL:
    return fabs(xi[0]) <= 1.0 + tol && fabs(xi[1]) <= 1.0 + tol;
  case TETRAHEDRON:
    return xi[0] >= -tol && xi[1] >= -tol && xi[2] >= -tol &&
           (1.0 - xi[0] - xi[1] - xi[2]) >= -tol;
  case HEXAHEDRON:
    return fabs(xi[0]) <= 1.0 + tol && fabs(xi[1]) <= 1.0 + tol && fabs(xi[2]) <= 1.0 + tol;
  default:
    return 0;
  }
}

/* Jacobian J[k][j] = d x_k / d xi_j of the vertex map */
static void vertex_jacobian(
    int nv, int dim, const double coords[][3], double dN[][3], double J[3][3]) {
  int i, j, k;
  for (k = 0; k < 3; k++)
    for (j = 0; j < 3; j++)
      J[k][j] = 0.0;
  for (i = 0; i < nv; i++)
    for (k = 0; k < dim; k++)
      for (j = 0; j < dim; j++)
        J[k][j] += coords[i][k] * dN[i][j];
}

static double det_dim(int dim, double J[3][3]) {
  if (dim == 2)
    return J[0][0] * J[1][1] - J[0][1] * J[1][0];
  return J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
         J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
         J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
}

/* Solve J dxi = r for dim 2 or 3 by Cramer's rule; returns -1 if singular. */
static int solve_dim(int dim, double J[3][3], const double r[3], double dxi[3]) {
  const double det = det_dim(dim, J);
  if (det == 0.0 || !isfinite(det))
    return -1;
  if (dim == 2) {
    dxi[0] = (r[0] * J[1][1] - J[0][1] * r[1]) / det;
    dxi[1] = (J[0][0] * r[1] - r[0] * J[1][0]) / det;
    dxi[2] = 0.0;
    return 0;
  }
  for (int c = 0; c < 3; c++) {
    double M[3][3];
    memcpy(M, J, sizeof(M));
    for (int k = 0; k < 3; k++)
      M[k][c] = r[k];
    dxi[c] = det_dim(3, M) / det;
  }
  return 0;
}

int cfd_dem_invert_map(int shape, const double coords[][3], const double x[3], double xi[3]) {
  const int nv = cfd_dem_num_vertices(shape);
  const int dim = cfd_dem_shape_dim(shape);
  double N[CFD_DEM_MAX_VERTS], dN[CFD_DEM_MAX_VERTS][3], J[3][3];
  double r[3], dxi[3], h = 0.0;
  int iter, i, k;

  if (nv < 0)
    return -1;

  /* length scale for the convergence test */
  for (i = 1; i < nv; i++) {
    double d2 = 0.0;
    for (k = 0; k < dim; k++)
      d2 += (coords[i][k] - coords[0][k]) * (coords[i][k] - coords[0][k]);
    h = fmax(h, sqrt(d2));
  }
  if (h == 0.0)
    return -1;

  for (iter = 0; iter < 50; iter++) {
    double rnorm = 0.0;
    cfd_dem_vertex_shape(shape, xi, N, dN);
    r[0] = r[1] = r[2] = 0.0;
    for (k = 0; k < dim; k++) {
      double xk = 0.0;
      for (i = 0; i < nv; i++)
        xk += N[i] * coords[i][k];
      r[k] = x[k] - xk;
      rnorm += r[k] * r[k];
    }
    if (sqrt(rnorm) <= 1.0e-12 * h)
      return 0;
    vertex_jacobian(nv, dim, coords, dN, J);
    if (solve_dim(dim, J, r, dxi) != 0)
      return -1;
    for (k = 0; k < dim; k++)
      xi[k] += dxi[k];
    /* a point far outside the element can send Newton off; that element is not it */
    for (k = 0; k < dim; k++)
      if (fabs(xi[k]) > 10.0)
        return -1;
  }
  return -1;
}

double cfd_dem_vertex_volumes(int shape, const double coords[][3], double vol[]) {
  const int nv = cfd_dem_num_vertices(shape);
  const int dim = cfd_dem_shape_dim(shape);
  double N[CFD_DEM_MAX_VERTS], dN[CFD_DEM_MAX_VERTS][3], J[3][3];
  double total = 0.0;
  int i;

  if (nv < 0)
    return 0.0;
  for (i = 0; i < nv; i++)
    vol[i] = 0.0;

  /* Gauss rules exact for the (multi)linear-times-Jacobian integrands */
  double pts[8][3], wts[8];
  int npts = 0;
  const double g = 1.0 / sqrt(3.0);
  switch (shape) {
  case QUADRILATERAL:
    for (int a = 0; a < 2; a++)
      for (int b = 0; b < 2; b++) {
        pts[npts][0] = a ? g : -g, pts[npts][1] = b ? g : -g, pts[npts][2] = 0.0;
        wts[npts++] = 1.0;
      }
    break;
  case HEXAHEDRON:
    for (int a = 0; a < 2; a++)
      for (int b = 0; b < 2; b++)
        for (int c = 0; c < 2; c++) {
          pts[npts][0] = a ? g : -g, pts[npts][1] = b ? g : -g, pts[npts][2] = c ? g : -g;
          wts[npts++] = 1.0;
        }
    break;
  case TRIANGLE:
    /* constant Jacobian: centroid rule is exact for linear N */
    pts[0][0] = pts[0][1] = 1.0 / 3.0, pts[0][2] = 0.0;
    wts[0] = 0.5;
    npts = 1;
    break;
  case TETRAHEDRON:
    pts[0][0] = pts[0][1] = pts[0][2] = 0.25;
    wts[0] = 1.0 / 6.0;
    npts = 1;
    break;
  default:
    return 0.0;
  }

  for (int q = 0; q < npts; q++) {
    cfd_dem_vertex_shape(shape, pts[q], N, dN);
    vertex_jacobian(nv, dim, coords, dN, J);
    const double detJ = fabs(det_dim(dim, J)) * wts[q];
    for (i = 0; i < nv; i++)
      vol[i] += N[i] * detJ;
    total += detJ;
  }
  return total;
}

double cfd_dem_drag_coefficient(int model, double rho, double mu, double eps, double d,
                                double umag) {
  switch (model) {
  case CFD_DEM_DRAG_STOKES:
    return 3.0 * M_PI * mu * d;

  case CFD_DEM_DRAG_DI_FELICE: {
    const double Re = rho * eps * d * umag / mu;
    double chi = 3.7;
    if (Re > 0.0) {
      const double l = 1.5 - log10(Re);
      chi = 3.7 - 0.65 * exp(-0.5 * l * l);
    }
    /* Cd |u| = (0.63 sqrt|u| + 4.8 sqrt(mu/(rho eps d)))^2, regular as |u| -> 0 */
    const double a = 0.63 * sqrt(umag) + 4.8 * sqrt(mu / (rho * eps * d));
    return 0.5 * a * a * rho * (M_PI * d * d / 4.0) * eps * eps * pow(eps, -chi);
  }

  default:
    return -1.0;
  }
}

int cfd_dem_drag_force(int model,
                       double rho,
                       double mu,
                       double eps,
                       double d,
                       const double u_rel[3],
                       int ncomp,
                       double F[3]) {
  double umag = 0.0;
  int k;

  F[0] = F[1] = F[2] = 0.0;
  for (k = 0; k < ncomp; k++)
    umag += u_rel[k] * u_rel[k];
  umag = sqrt(umag);

  const double coef = cfd_dem_drag_coefficient(model, rho, mu, eps, d, umag);
  if (coef < 0.0)
    return -1;
  for (k = 0; k < ncomp; k++)
    F[k] = coef * u_rel[k];
  return 0;
}
