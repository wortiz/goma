/*
 * cuda_structs.cuh
 *
 *  Created on: Aug 9, 2016
 *      Author: wortiz
 */

#ifndef CUDA_STRUCTS_CUH_
#define CUDA_STRUCTS_CUH_

#include "cuda_constants.cuh"
#include <Epetra_config.h>
#include <Epetra_MpiComm.h>

namespace cuda {

typedef struct {
  double wt; /* Gauss weight. */
  double x[CUDA_DIM]; /* Position in physical space. */
  /*
   * Add some useful quantities for curvilinear orthogonal coordinate
   * systems...note the difference between raw derivatives and the gradient
   * operator...(see mm_fill_aux.c for explanations of each of these variables)
   */
  double h[CUDA_DIM]; /* Scale factors. */

  double h3; /* Volume element factor. */
  double v[CUDA_DIM]; /* Velocity. */
  double xi[CUDA_DIM];
  double P; /* Pressure. */
  /*
   * Grads of scalars...
   */

  double grad_P[CUDA_DIM]; /* Gradient of pressure. */

  /*
   * Grads of vectors...
   */

  double div_v; /* Divergence of velocity. */
  double grad_v[CUDA_DIM][CUDA_DIM]; /* Gradient of velocity.  d (v_i) / d (x_j) */
} field_variables;

typedef struct {
  double v[CUDA_DIM][CUDA_MDE];
  double P[CUDA_MDE];
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
  double phi[CUDA_MDE]; /* phi_i */
  double dphidxi[CUDA_MDE][CUDA_DIM]; /* d(phi_i)/d(xi_j) */

  /*
   * beer_belly() fills in these elemental Jacobian things...
   */
  double J[CUDA_DIM][CUDA_DIM];
  /*
   *  determinant of the jacobian of the matrix transformation
   *  of the ShapeVar shape function.
   */
  double detJ;
  double B[CUDA_DIM][CUDA_DIM]; /* inverse Jacobian */
  double d_det_J_dm[CUDA_DIM][CUDA_MDE];
  double dJ[CUDA_DIM][CUDA_DIM][CUDA_DIM][CUDA_MDE]; /* d( J[i][j] ) / d (d_k,l) */
  double dB[CUDA_DIM][CUDA_DIM][CUDA_DIM][CUDA_MDE];

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

  double d_phi[CUDA_MDE][CUDA_DIM]; /* d_phi[i][a]    = d(phi_i)/d(q_a) */
  double grad_phi[CUDA_MDE][CUDA_DIM]; /* grad_phi[i][a] = e_a . grad(phi_i) */

  double grad_phi_e[CUDA_MDE][CUDA_DIM][CUDA_DIM][CUDA_DIM]; /* grad_phi_e[i][a][p][q] */
  /* = (e_p e_q): grad(phi_i e_a) */
} basis_function;

typedef struct {
  double v[2][2][2][9];
  double P[2][2][9];
} stress_dependence;

typedef struct {
  double J[CUDA_NUM_EQNS][CUDA_NUM_EQNS][CUDA_MDE][CUDA_MDE];
  double R[CUDA_NUM_EQNS][CUDA_MDE];
} local_element_contributions;

typedef struct {
  int dirichlet_index;
  double value;
  boundary_condition_type type;
  equation eqn;
} boundary_condition;

typedef struct {
  int id;
  boundary_condition bcs[MAX_BCS];
  int num_bcs;
  int gnn[CUDA_MDE];
  int eqn_index[CUDA_NUM_EQNS];
  int eqn_dof[CUDA_NUM_EQNS];
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
  double *delta_x;
  double *b;

  int **solution_index;
} problem_data;

} // namespace cuda
#endif /* CUDA_STRUCTS_CUH_ */
