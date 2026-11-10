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
 * Mesh-independent helpers for CFD-DEM coupling: linear (vertex) shape functions
 * used for particle location and particle-to-node deposition, and drag correlations.
 *
 * Deposition uses the element *vertex* (Q1/P1) functions even when the fluid fields are
 * quadratic, because quadratic corner functions go negative inside the element (and the
 * serendipity corner integrals are negative), which would produce eps > 1 or negative
 * lumped volumes.  Vertex functions are non-negative and sum to one, so the deposition
 * is conservative: sum_i w_i = 1 for every particle.
 */

#ifndef GOMA_CFD_DEM_UTIL_H
#define GOMA_CFD_DEM_UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

#define CFD_DEM_MAX_VERTS 8

/* Number of vertices for an element shape (TRIANGLE, QUADRILATERAL, TETRAHEDRON,
 * HEXAHEDRON from el_elm.h), or -1 if unsupported. */
int cfd_dem_num_vertices(int shape);

/* Spatial dimension of a shape, or -1 if unsupported. */
int cfd_dem_shape_dim(int shape);

/* Vertex shape functions N[i] and parametric derivatives dN[i][k] = dN_i/dxi_k at xi.
 * Quad/hex use xi in [-1,1]^d with exodus vertex ordering; tri/tet use area/volume
 * coordinates xi in the unit simplex.  Returns 0, or -1 for an unsupported shape. */
int cfd_dem_vertex_shape(int shape, const double xi[3], double N[], double dN[][3]);

/* Newton inversion of the vertex map: find xi with x(xi) = x.  coords[i][k] holds
 * vertex i coordinate k.  Returns 0 on convergence, -1 otherwise.  xi is the starting
 * guess on input (pass the element center if unknown). */
int cfd_dem_invert_map(int shape, const double coords[][3], const double x[3], double xi[3]);

/* Parametric center of the element (start guess for cfd_dem_invert_map). */
void cfd_dem_shape_center(int shape, double xi[3]);

/* True if xi lies in the parent element, with tolerance tol. */
int cfd_dem_xi_inside(int shape, const double xi[3], double tol);

/* Lumped vertex volumes vol[i] = integral of N_i over the element (all > 0 for a valid
 * element).  Returns the total element volume. */
double cfd_dem_vertex_volumes(int shape, const double coords[][3], double vol[]);

/* Drag coefficient beta with F = beta * (u - v), evaluated at slip speed umag.
 * Returns a negative value for an unknown model.  See cfd_dem_drag_force. */
double cfd_dem_drag_coefficient(int model, double rho, double mu, double eps, double d,
                                double umag);

/* Drag force on a particle of diameter d moving at v in fluid moving at u, written in
 * terms of u_rel = u - v.  F points along u_rel.  eps is the local fluid fraction.
 *  CFD_DEM_DRAG_DI_FELICE: Di Felice (1994), Int. J. Multiphase Flow 20, 153-159
 *      F = 1/2 Cd rho (pi d^2/4) eps^2 |u_rel| u_rel eps^(-chi)
 *      Cd = (0.63 + 4.8/sqrt(Re))^2, Re = rho eps d |u_rel| / mu
 *      chi = 3.7 - 0.65 exp(-(1.5 - log10 Re)^2 / 2)
 *  CFD_DEM_DRAG_STOKES: F = 3 pi mu d u_rel  (dilute, for verification)
 * ncomp is the number of velocity components (2 or 3).  Returns 0, or -1 if the model
 * is unknown. */
int cfd_dem_drag_force(int model,
                       double rho,
                       double mu,
                       double eps,
                       double d,
                       const double u_rel[3],
                       int ncomp,
                       double F[3]);

#ifdef __cplusplus
}
#endif

#endif /* GOMA_CFD_DEM_UTIL_H */
