/*
 * ns_structs.h
 *
 *  Created on: Jun 30, 2016
 *      Author: wortiz
 */

#ifndef NS_STRUCTS_H_
#define NS_STRUCTS_H_

#include "ns_constants.h"
#include <Epetra_config.h>
#include <Epetra_MpiComm.h>
extern "C" {
#include "exo_read_mesh.h"
}

typedef struct {
  double wt; /* Gauss weight. */
  double x[DIM]; /* Position in physical space. */
  /*
   * Add some useful quantities for curvilinear orthogonal coordinate
   * systems...note the difference between raw derivatives and the gradient
   * operator...(see mm_fill_aux.c for explanations of each of these variables)
   */
  double h[DIM]; /* Scale factors. */

  double h3; /* Volume element factor. */
  double v[DIM]; /* Velocity. */
  double v_star[DIM]; /* Velocity. */
  double xi[DIM];
  double P; /* Pressure. */
  double P_star; /* Pressure. */
  /*
   * Grads of scalars...
   */

  double grad_P[DIM]; /* Gradient of pressure. */
  double grad_P_star[DIM]; /* Gradient of pressure. */

  /*
   * Grads of vectors...
   */

  double div_v; /* Divergence of velocity. */
  double grad_v[DIM][DIM]; /* Gradient of velocity.  d (v_i) / d (x_j) */
  double div_v_star; /* Divergence of velocity. */
  double grad_v_star[DIM][DIM]; /* Gradient of velocity.  d (v_i) / d (x_j) */

} field_variables;

typedef struct {
  double v[DIM][MDE];
  double P[MDE];
} element_stiffness_pointers;

typedef struct {
  //int ielem_type;		/* old SHM identifier of elements... */
  int interpolation; /* eg., I_Q1, ... */
  int element_shape; /* eg., QUADRILATERAL, ...*/
  int shape;         /* e.g. BIQUAD_QUAD */
  int num_dof; /* How many degrees of freedom are involved
   * in the interpolation of this element? */
  /*
   * load_basis_functions() fills in this stuff...
   */
  double phi[MDE]; /* phi_i */
  double dphidxi[MDE][DIM]; /* d(phi_i)/d(xi_j) */

  /*
   * beer_belly() fills in these elemental Jacobian things...
   */
  double J[DIM][DIM];
  /*
   *  determinant of the jacobian of the matrix transformation
   *  of the ShapeVar shape function.
   */
  double detJ;
  double B[DIM][DIM]; /* inverse Jacobian */
  double d_det_J_dm[DIM][MDE];
  double dJ[DIM][DIM][DIM][MDE]; /* d( J[i][j] ) / d (d_k,l) */
  double dB[DIM][DIM][DIM][MDE];

  /*
   * These two things are the same in Cartesian coordinates, but not
   * in nonCartesian coordinate systems with nontrivial scale factors
   * and spatially-varying unit vectors...
   *
   * Strictly, e_a . grad(phi_i) =    1    d ( phi[i] )
   *				   ------  ------------
   *				    h[a]   d ( x_a )
   * where:
   *		h[a] == scale factors
   *		x_a  == physical coordinates (eg., z,r,theta)
   *
   *
   * Thus, there are two transformations...
   *
   *	  d phi[i]             d phi[i]		  1    d phi[i]
   *      --------    ---->    --------   ----> -----  --------
   *      d xi[j]	       d x[j]		 h[j]  d x[j]
   *
   *		    elemental		  scale
   *		    Jacobian		  factors
   */

  double d_phi[MDE][DIM]; /* d_phi[i][a]    = d(phi_i)/d(q_a) */
  double grad_phi[MDE][DIM]; /* grad_phi[i][a] = e_a . grad(phi_i) */

  double grad_phi_e[MDE][DIM][DIM][DIM]; /* grad_phi_e[i][a][p][q] */
  /* = (e_p e_q): grad(phi_i e_a) */
} basis_function;

typedef struct {
  double v[2][2][2][9];
  double P[2][2][9];
} stress_dependence;

typedef struct {
  double J[2][2][MDE][MDE];
  double R[2][MDE];
} local_element_contributions;

typedef struct {
  int dirichlet_index;
  double value;
  boundary_condition_type type;
  equation eqn;
} boundary_condition;

#define MAX_BFS 2

typedef struct {
  int id;
  boundary_condition bcs[MAX_BCS];
  int num_bcs;
  int gnn[MDE];
  int eqn_index[NUM_EQNS];
  int eqn_dof[NUM_EQNS];
} element;


typedef struct {
  int *rowptr;
  int *colind;
  double *val;
  int m;
  int nnz;
} CrsMatrix;

typedef struct {
  CrsMatrix *A;
  double *x;
  double *x_old;
  double *xdot;
  double *xdot_old;
  double *delta_x;
  double *b;
  int num_eqns;
  int eqn_index[NUM_EQNS];
  int inv_eqn_index[NUM_EQNS];
  int **solution_index;
} matrix_data;

typedef struct {
  mesh_data *mesh;
  element *elements;
  matrix_data **systems;
  boundary_condition *bcs;
  field_variables *fvs;
  field_variables *fv_olds;
  int num_elements;
  int num_matrices;
  int num_bcs;
} problem_data;
  
#endif /* NS_STRUCTS_H_ */
