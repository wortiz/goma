/*
 * solve_nonlinear.cu
 *
 *  Created on: Jul 5, 2016
 *      Author: wortiz
 */

#include <stdio.h>
#include <string>
#include <chrono>
#include <ctime>
#include <iostream>
#include <assert.h>
#include "cuda_macros.h"

extern "C" {
#include "exo_read_mesh.h"
}
#include "cuda_interface.h"
#include "cuda_solve_nonlinear.cuh"
//#include "ns_basis_functions.cuh"
//#include "ns_assemble.cuh"
#include <iostream>

#include "umfpack.h"

void cuda_trilinos_solve(const Epetra_Comm & comm, cuda::problem_data *mat_system);

namespace cuda {

__device__ double atomicAdd_double(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__host__ __device__ int in_list(int start, int end, int value, int *list)
{
  int low = start;
  int high = end;

  while (low <= high) {
    int mid = (low+high) / 2;
    if (list[mid] == value) {
      return mid;
    } else if (list[mid] < value) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return -1;
}

__host__ __device__ int index_solution(problem_data *mat_system, int node, int eqn) {
  return mat_system->solution_index[eqn][node];
}

__host__ __device__ void row_sum_scale(problem_data *mat_system) {
  double scale;

  for (int i = 0; i < mat_system->A->m; i++) {
    scale = 0;
    int start = mat_system->A->rowptr[i];
    int end = mat_system->A->rowptr[i+1];
    for (int j = start; j < end; j++) {
      scale += mat_system->A->val[j];
    }

    if (fabs(scale) > 1e-14) {
      scale = 1.0 / scale;
      mat_system->b[i] *= scale;
      for (int j = start; j < end; j++) {
	mat_system->A->val[j] *= scale;
      }
    }
  }
}

__host__ __device__ void load_fv(problem_data *mat_system, field_variables *fv,
				 basis_function **bfs, element *elem) {

  fv->h[0] = 1;
  fv->h[1] = 1;
  fv->h3 = 1;

  int v, a;
  int idx;
  for (v = CUDA_VELOCITY1, a = 0; v <= CUDA_VELOCITY2; v++, a++) {
    fv->v[a] = 0.0;

    for (int i = 0; i < bfs[v]->num_dof; i++) {
      idx = index_solution(mat_system, elem->gnn[i] - 1, v);
      fv->v[a] += mat_system->x[idx] * bfs[v]->phi[i];
    }
  }

  fv->P = 0;

  v = CUDA_PRESSURE;
  for (int i = 0; i < bfs[CUDA_PRESSURE]->num_dof; i++) {
    idx = index_solution(mat_system, elem->gnn[i] - 1, v);
    fv->P += mat_system->x[idx] * bfs[CUDA_PRESSURE]->phi[i];
  }
}

__host__ __device__ void load_fv_grad(problem_data *mat_system, field_variables *fv,
				      basis_function **bfs, element *elem) {
  int p, q, r, i;
  int v;
  int dofs;

  v = CUDA_VELOCITY1;
  dofs = bfs[v]->num_dof;
  for (p = 0; p < CUDA_DIM; p++) {
    for (q = 0; q < CUDA_DIM; q++) {
      fv->grad_v[p][q] = 0.0;
      for (r = 0; r < CUDA_DIM; r++) {
	for (i = 0; i < dofs; i++) {
	  int idx = index_solution(mat_system, elem->gnn[i] - 1, v+r);
	  fv->grad_v[p][q] += mat_system->x[idx]
	    * bfs[v]->grad_phi_e[i][r][p][q];
	}
      }
    }
  }

  fv->div_v = fv->grad_v[0][0] + fv->grad_v[1][1];

  v = CUDA_PRESSURE;
  dofs = bfs[v]->num_dof;
  for (p = 0; p < CUDA_DIM; p++) {
    fv->grad_P[p] = 0.0;

    for (i = 0; i < dofs; i++) {
      int idx = index_solution(mat_system, elem->gnn[i] - 1, v);
      fv->grad_P[p] += mat_system->x[idx] * bfs[v]->grad_phi[i][p];
    }
  }

}

__host__ __device__ void set_dirichlet(mesh_data *mesh, local_element_contributions *lec,
				       element elem) {
  for (int i = 0; i < elem.num_bcs; i++) {
    int eqn = elem.bcs[i].eqn;
    for (int j = 0; j < CUDA_MDE; j++) {
      int node = elem.gnn[j];
      int didx;
      for (didx = 0; didx < mesh->num_dirichlet; didx++) {
	int id = elem.bcs[i].dirichlet_index;
	if (id == mesh->dirichlet_ids[didx]) {
	  break;
	}
      }
      for (int idx = mesh->dirichlet_node_index[didx];
	   idx < mesh->dirichlet_node_index[didx + 1]; idx++) {
	if (node == mesh->dirichlet_node_list[idx]) {
	  int b, d;

	  for (b = 0; b < CUDA_NUM_EQNS; b++) {
	    for (d = 0; d < CUDA_MDE; d++) {
	      lec->J[eqn][b][j][d] = 0.0;
	    }

	  }
	  lec->J[eqn][eqn][j][j] = 1;

	  lec->R[eqn][j] = 0.0;
	}
      }
    }
  }
}

__device__ void load_lec_gpu(local_element_contributions *lec, mesh_data *mesh,
				  basis_function **bfs, element elem, problem_data *mat_system) {

  for (int e = 0; e < CUDA_NUM_EQNS; e++) {
    for (int i = 0; i < bfs[e]->num_dof; i++) {
      int row = index_solution(mat_system, elem.gnn[i] - 1, e);
      assert(row != -1);
      atomicAdd_double(&(mat_system->b[row]), lec->R[e][i]);
      for (int v = 0; v < CUDA_NUM_EQNS; v++) {
	for (int j = 0; j < bfs[v]->num_dof; j++) {
	  int col = index_solution(mat_system, elem.gnn[j] - 1, v);
	  assert(col != -1);
	  int index = in_list(mat_system->A->rowptr[row],
			      mat_system->A->rowptr[row+1],
			      col,
			      mat_system->A->colind);
	  assert(index != -1);

	  atomicAdd_double(&(mat_system->A->val[index]), lec->J[e][v][i][j]);
	}
      }
    }
  }
}

__host__ __device__ void load_lec(local_element_contributions *lec, mesh_data *mesh,
				  basis_function **bfs, element elem, problem_data *mat_system) {

  for (int e = 0; e < CUDA_NUM_EQNS; e++) {
    for (int i = 0; i < bfs[e]->num_dof; i++) {
      int row = index_solution(mat_system, elem.gnn[i] - 1, e);
      assert(row != -1);
      mat_system->b[row] += lec->R[e][i];

      for (int v = 0; v < CUDA_NUM_EQNS; v++) {
	for (int j = 0; j < bfs[v]->num_dof; j++) {
	  int col = index_solution(mat_system, elem.gnn[j] - 1, v);
	  assert(col != -1);
	  int index = in_list(mat_system->A->rowptr[row],
			      mat_system->A->rowptr[row+1],
			      col,
			      mat_system->A->colind);
	  assert(index != -1);

	  mat_system->A->val[index] += lec->J[e][v][i][j];
	}
      }
    }
  }
}

__device__ void zero_lec(local_element_contributions *lec)
{
  for (int e = 0; e < CUDA_NUM_EQNS; e++) {
    for (int i = 0; i < CUDA_MDE; i++) {
      lec->R[e][i] = 0;
    }

    for (int v = 0; v < CUDA_NUM_EQNS; v++) {
      for (int i = 0; i < CUDA_MDE; i++) {
	for (int j = 0; j < CUDA_MDE; j++) {
	  lec->J[e][v][i][j] = 0;
	}
      }
    }
  }
}

__host__ __device__ double find_gauss_weight(basis_function *bf, int ip)
{
  const double ftemp1 = 0.55555555555555555556;
  const double ftemp2 = 0.88888888888888888888;
  double weight = 0;
  double weight_s, weight_t;
  switch (bf->element_shape) {
  case BILINEAR_QUAD: /* bilinear quadrilateral */
    weight = 1.0;
    break;
  case BIQUAD_QUAD:
    if (ip % 3 == 0) {
      weight_s = ftemp1;
    } else if ((ip - 1) % 3 == 0) {
      weight_s = ftemp2;
    } else {
      weight_s = ftemp1;
    }

    if (ip < 3) {
      weight_t = ftemp1;
    } else if (ip < 6) {
      weight_t = ftemp2;
    } else {
      weight_t = ftemp1;
    }
    weight = weight_s * weight_t;
    break;
  }

  return weight;
}

__host__ __device__ void find_gauss_points(double *xi, basis_function *bf,
                                           int ip)
{
  const double recip_root_three = 0.57735026918962584208;
  const double sqrt_three_fifths = 0.77459666924148340428;

  switch (bf->element_shape) {
  case BILINEAR_QUAD: /* bilinear quadrilateral */
    xi[0] = (ip % 2 == 0) ? recip_root_three : -recip_root_three;
    xi[1] = (ip < 2) ? recip_root_three : -recip_root_three;
    xi[2] = 0.0;
    break;
  case BIQUAD_QUAD:
    if (ip % 3 == 0) {
      xi[0] = sqrt_three_fifths;
    } else if ((ip - 1) % 3 == 0) {
      xi[0] = 0.0;
    } else {
      xi[0] = -sqrt_three_fifths;
    }
    if (ip < 3) {
      xi[1] = sqrt_three_fifths;
    } else if (ip < 6) {
      xi[1] = 0.0;
    } else {
      xi[1] = -sqrt_three_fifths;
    }
    break;
  }
}

__host__ __device__ double shape_function(double *xi, basis_function *bf,
                                          int dof)
{
  double s = xi[0];
  double t = xi[1];
  double value = 0;
  switch (bf->shape) {
  case BILINEAR_QUAD:
    switch (dof) {
    case 0:
      value = 0.25 * (1.0 - s) * (1.0 - t);
      break;
    case 1:
      value = 0.25 * (1.0 + s) * (1.0 - t);
      break;
    case 2:
      value = 0.25 * (1.0 + s) * (1.0 + t);
      break;
    case 3:
      value = 0.25 * (1.0 - s) * (1.0 + t);
      break;
    default:
      break;
    }
    break;
  case BIQUAD_QUAD:
    switch (dof) {
    case 0:
      value = .25 * (1.0 - s) * (1.0 - t) * (-s - t - 1.0)
	+ .25 * (1.0 - s * s) * (1.0 - t * t);
      break;
    case 1:
      value = .25 * (1.0 + s) * (1.0 - t) * (s - t - 1.0)
	+ .25 * (1.0 - s * s) * (1.0 - t * t);
      break;
    case 2:
      value = .25 * (1.0 + s) * (1.0 + t) * (s + t - 1.0)
	+ .25 * (1.0 - s * s) * (1.0 - t * t);
      break;
    case 3:
      value = .25 * (1.0 - s) * (1.0 + t) * (-s + t - 1.0)
	+ .25 * (1.0 - s * s) * (1.0 - t * t);
      break;
    case 4:
      value = .5 * (1.0 - s * s) * (1.0 - t)
	- .5 * (1.0 - s * s) * (1.0 - t * t);
      break;
    case 5:
      value = .5 * (1.0 + s) * (1.0 - t * t)
	- .5 * (1.0 - s * s) * (1.0 - t * t);
      break;
    case 6:
      value = .5 * (1.0 - s * s) * (1.0 + t)
	- .5 * (1.0 - s * s) * (1.0 - t * t);
      break;
    case 7:
      value = .5 * (1.0 - s) * (1.0 - t * t)
	- .5 * (1.0 - s * s) * (1.0 - t * t);
      break;
    case 8:
      value = (1.0 - s * s) * (1.0 - t * t);
      break;
    default:
      break;
    }
    break;
  default:
    break;

  }

  return value;
}

__host__ __device__ double shape_function_derivative(double *xi,
                                                     basis_function *bf,
                                                     int dof, int direction)
{
  double s = xi[0];
  double t = xi[1];
  double value = 0;
  switch (bf->shape) {
  case BILINEAR_QUAD:
    switch (direction) {
    case 0:  // s direction
      switch (dof) {
      case 0:
	value = -0.25 * (1.0 - t);
	break;
      case 1:
	value = 0.25 * (1.0 - t);
	break;
      case 2:
	value = 0.25 * (1.0 + t);
	break;
      case 3:
	value = -0.25 * (1.0 + t);
	break;
      }
      break;
    case 1:  // t direction
      switch (dof) {
      case 0:
	value = -0.25 * (1.0 - s);
	break;
      case 1:
	value = -0.25 * (1.0 + s);
	break;
      case 2:
	value = 0.25 * (1.0 + s);
	break;
      case 3:
	value = 0.25 * (1.0 - s);
	break;
      }
      break;
    }

    break;

  case BIQUAD_QUAD:
    switch (direction) {
    case 0:  // s direction
      switch (dof) {
      case 0:
	value = -.25 * (1.0 - t) * (-2.0 * s - t) - 0.5 * s * (1 - t * t);
	break;
      case 1:
	value = .25 * (1.0 - t) * (2.0 * s - t) - 0.5 * s * (1 - t * t);
	break;
      case 2:
	value = .25 * (1.0 + t) * (2.0 * s + t) - 0.5 * s * (1 - t * t);
	break;
      case 3:
	value = -.25 * (1.0 + t) * (-2.0 * s + t) - 0.5 * s * (1 - t * t);
	break;
      case 4:
	value = -s * (1.0 - t) + s * (1 - t * t);
	break;
      case 5:
	value = .5 * (1.0 - t * t) + s * (1 - t * t);
	break;
      case 6:
	value = -s * (1.0 + t) + s * (1 - t * t);
	break;
      case 7:
	value = -.5 * (1.0 - t * t) + s * (1 - t * t);
	break;
      case 8:
	value = -2.0 * s * (1.0 - t * t);
	break;
      }

      break;

    case 1:  // t direction
      switch (dof) {
      case 0:
	value = -.25 * (1.0 - s) * (-s - 2.0 * t)
	  - 0.5 * t * (1.0 - s * s);
	break;
      case 1:
	value = -.25 * (1.0 + s) * (s - 2.0 * t)
	  - 0.5 * t * (1.0 - s * s);
	break;
      case 2:
	value = .25 * (1.0 + s) * (s + 2.0 * t) - 0.5 * t * (1.0 - s * s);
	break;
      case 3:
	value = .25 * (1.0 - s) * (-s + 2.0 * t)
	  - 0.5 * t * (1.0 - s * s);
	break;
      case 4:
	value = -.5 * (1.0 - s * s) + t * (1.0 - s * s);
	break;
      case 5:
	value = -(1.0 + s) * t + t * (1.0 - s * s);
	break;
      case 6:
	value = .5 * (1.0 - s * s) + t * (1.0 - s * s);
	break;
      case 7:
	value = -(1.0 - s) * t + t * (1.0 - s * s);
	break;
      case 8:
	value = -2.0 * t * (1.0 - s * s);
	break;
      }
      break;
    }

    break;

  default:

    break;

  }

  return value;
}

__host__ __device__ void jacobian_inverse_and_mapping(mesh_data *mesh, field_variables *fv,
                                                      basis_function **bfs,
                                                      element elem)
{
  int i;
  int j;
  int k;
  int dim = 2;

  basis_function *shape_bf = bfs[CUDA_VELOCITY1];

  // Mapping
  for (i = 0; i < dim; i++) {
    fv->x[i] = 0;
    for (j = 0; j < shape_bf->num_dof; j++) {

      fv->x[i] += mesh->coord[i][elem.gnn[j]-1] * shape_bf->phi[j];

    }
  }

  // Jacobian
  for (i = 0; i < dim; i++) {
    for (j = 0; j < dim; j++) {
      shape_bf->J[i][j] = 0;
      for (k = 0; k < shape_bf->num_dof; k++) {

        shape_bf->J[i][j] += mesh->coord[j][elem.gnn[k]-1] * shape_bf->dphidxi[k][i];
      }
    }
  }

  shape_bf->detJ = shape_bf->J[0][0] * shape_bf->J[1][1]
    - shape_bf->J[0][1] * shape_bf->J[1][0];

  shape_bf->B[0][0] = shape_bf->J[1][1] / shape_bf->detJ;
  shape_bf->B[0][1] = -shape_bf->J[0][1] / shape_bf->detJ;
  shape_bf->B[1][0] = -shape_bf->J[1][0] / shape_bf->detJ;
  shape_bf->B[1][1] = shape_bf->J[0][0] / shape_bf->detJ;

  /* Copy basis function */

  for (int i = 0; i < CUDA_NUM_EQNS; i++) {
    if (i == CUDA_VELOCITY1)
      continue;
    bfs[i]->J[0][0] = shape_bf->J[0][0];
    bfs[i]->J[0][1] = shape_bf->J[0][1];
    bfs[i]->J[1][0] = shape_bf->J[1][0];
    bfs[i]->J[1][1] = shape_bf->J[1][1];

    bfs[i]->B[0][0] = shape_bf->B[0][0];
    bfs[i]->B[0][1] = shape_bf->B[0][1];
    bfs[i]->B[1][0] = shape_bf->B[1][0];
    bfs[i]->B[1][1] = shape_bf->B[1][1];

    bfs[i]->detJ = shape_bf->detJ;
  }
}

__host__ __device__ void load_basis_functions(int dim, double *xi,
                                              basis_function *bf)
{
  // Assume 2d atm
  for (int i = 0; i < bf->num_dof; i++) {
    bf->phi[i] = shape_function(xi, bf, i);
    for (int j = 0; j < dim; j++) {
      bf->dphidxi[i][j] = shape_function_derivative(xi, bf, i, j);
    }
  }
}
__host__ __device__ void load_basis_functions_grad(field_variables *fv,
                                                   basis_function *bfv)
{
  int i, p, a, q;
  int dofs;
  dofs = bfv->num_dof;

  for (i = 0; i < dofs; i++) {
    bfv->d_phi[i][0] = (bfv->B[0][0] * bfv->dphidxi[i][0]
			+ bfv->B[0][1] * bfv->dphidxi[i][1]);
    bfv->d_phi[i][1] = (bfv->B[1][0] * bfv->dphidxi[i][0]
			+ bfv->B[1][1] * bfv->dphidxi[i][1]);
  }

  for (i = 0; i < dofs; i++) {
    for (p = 0; p < CUDA_DIM; p++) {
      bfv->grad_phi[i][p] = (bfv->d_phi[i][p]) / (fv->h[p]);
    }
  }

  for (i = 0; i < dofs; i++) {
    for (p = 0; p < CUDA_DIM; p++) {
      for (a = 0; a < CUDA_DIM; a++) {
        for (q = 0; q < CUDA_DIM; q++) {
          if (q == a)
            bfv->grad_phi_e[i][a][p][a] = bfv->grad_phi[i][p];
          else
            bfv->grad_phi_e[i][a][p][q] = 0.0;
        }
      }
    }
  }
}

__host__ __device__ void fluid_stress_light(int dim, field_variables *fv,
                                            basis_function **bf, int *eqn_index,
                                            int *eqn_dof, double Pi[CUDA_DIM][CUDA_DIM],
                                            stress_dependence *d_Pi, double mu);

/* assemble_momentum -- assemble terms (Residual &| Jacobian) for momentum eqns
 *
 * in:
 * 	ei -- pointer to Element Indeces	structure
 *	pd -- pointer to Problem Description	structure
 *	af -- pointer to Action Flag		structure
 *	bf -- pointer to Basis Function		structure
 *	fv -- pointer to Field Variable		structure
 *	cr -- pointer to Constitutive Relation	structure
 *	md -- pointer to Mesh Derivative	structure
 *	me -- pointer to Material Entity	structure
 * 	ija -- vector of pointers into the a matrix
 * 	a  -- global Jacobian matrix
 * 	R  -- global residual vector
 *
 * out:
 *	a   -- gets loaded up with proper contribution
 *	lec -- gets loaded up with local contributions to resid, Jacobian
 * 	r  -- residual RHS vector
 *
 * Created:	Wed Dec  8 14:03:06 MST 1993 pasacki@sandia.gov
 *
 * Revised:	Sun Feb 27 06:53:12 MST 1994 pasacki@sandia.gov
 *
 *
 * Note: currently we do a "double load" into the addresses in the global
 *       "a" matrix and resid vector held in "esp", as well as into the
 *       local accumulators in "lec".
 *
 */

__host__ __device__ int assemble_momentum_light(
						int dim, field_variables *fv, basis_function **bf,
						local_element_contributions *lec, int *eqn_index, int *eqn_dof, double time,
						double tt, double dt, double mu, double rho)
{

  //! dim is the length of the velocity vector

  int i, j, p, q, a, b;

  int eqn, var, peqn, pvar;

  int status;

  double h3; /* Volume element (scale factors). */

  /* field variables */
  /*double grad_v[CUDA_DIM][CUDA_DIM];*//* Gradient of v. */
  double *grad_v[CUDA_DIM];
  double *v = fv->v;

  double det_J; /* determinant of element Jacobian */

  double d_area;

  double advection;
  double advection_a, advection_b;
  double diffusion;
  basis_function * bfm;

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
  double phi_i;
  double (*grad_phi_i_e_a)[CUDA_DIM] = NULL;
  /*
   * Interpolation functions for variables and some of their derivatives.
   */

  double phi_j;

  double * phi_i_vector, *phi_j_vector;

  double Pi[CUDA_DIM][CUDA_DIM];
  stress_dependence d_Pi_struct;
  stress_dependence *d_Pi = &d_Pi_struct;

  double wt;

  /* coefficient variables for the Brinkman Equation
     in flows through porous media: KSC on 5/10/95 */

  double *J = NULL;

  /*
   * Petrov-Galerkin weighting functions for i-th residuals
   * and some of their derivatives...
   */

  double wt_func;

  status = 0;

  /*
   * Unpack variables from structures for local convenience...
   */

  eqn = CUDA_VELOCITY1;

  wt = fv->wt;

  det_J = bf[eqn]->detJ; /* Really, ought to be mesh eqn. */

  h3 = fv->h3; /* Differential volume element (scales). */

  d_area = det_J * wt * h3;

  for (a = 0; a < dim; a++)
    grad_v[a] = fv->grad_v[a];

  /*
   * Calculate the momentum stress tensor at the current gauss point
   */
  fluid_stress_light(dim, fv, bf, eqn_index, eqn_dof, Pi, d_Pi, mu);
  /*
   * Residuals_________________________________________________________________
   */

  /*
   * Assemble each component "a" of the momentum equation...
   */
  for (a = 0; a < dim; a++) {
    eqn = CUDA_VELOCITY1 + a;
    peqn = eqn_index[eqn];
    bfm = bf[eqn];

    /*
     * In the element, there will be contributions to this many equations
     * based on the number of degrees of freedom...
     */

    phi_i_vector = bfm->phi;

    for (i = 0; i < eqn_dof[eqn]; i++) {

      phi_i = phi_i_vector[i];
      grad_phi_i_e_a = bfm->grad_phi_e[i][a];

      /* only use Petrov Galerkin on advective term - if required */
      wt_func = phi_i;
      /* add Petrov-Galerkin terms as necessary */

      advection = 0.;

      for (p = 0; p < dim; p++) {
        advection += v[p] * grad_v[p][a];
      }
      advection *= rho;
      advection *= -wt_func * d_area;

      diffusion = 0.;
      for (p = 0; p < dim; p++) {
        for (q = 0; q < dim; q++) {
          diffusion += grad_phi_i_e_a[p][q] * Pi[q][p];
        }
      }

      diffusion *= -d_area;

      /*
       * Add contributions to this residual (globally into Resid, and
       * locally into an accumulator)
       */

      /*lec->R[peqn][ii] += mass + advection + porous + diffusion + source;*/
      lec->R[peqn][i] += advection + diffusion;
    } /* end of for (i=0,eqn_dofs...) */
  }

  /*
   * Jacobian terms...
   */

  for (a = 0; a < dim; a++) {
    eqn = CUDA_VELOCITY1 + a;
    peqn = eqn_index[eqn];
    bfm = bf[eqn];

    phi_i_vector = bfm->phi;

    for (i = 0; i < eqn_dof[eqn]; i++) {

      phi_i = phi_i_vector[i];

      /* Assign pointers into the bf structure */

      grad_phi_i_e_a = bfm->grad_phi_e[i][a];

      wt_func = phi_i;
      /* add Petrov-Galerkin terms as necessary */

      /*
       * J_m_v
       */
      for (b = 0; b < dim; b++) {
        var = CUDA_VELOCITY1 + b;
        pvar = eqn_index[var];

        J = lec->J[peqn][pvar][i];

        phi_j_vector = bf[var]->phi;

        for (j = 0; j < eqn_dof[var]; j++) {

          phi_j = phi_j_vector[j];

          advection = 0.;
          advection_a = 0.;

          advection_a += phi_j * grad_v[b][a];

          for (p = 0; p < dim; p++) {
            advection_a += v[p] * bf[var]->grad_phi_e[j][b][p][a];
          }

          advection_a *= rho;
          advection_b = 0.;

          advection_a *= -wt_func * d_area;
          advection = advection_a + advection_b;

          diffusion = 0.;

          for (p = 0; p < dim; p++) {
            for (q = 0; q < dim; q++) {
              diffusion += grad_phi_i_e_a[p][q] * d_Pi->v[q][p][b][j];
            }
          }

          diffusion *= -d_area;

          J[j] += advection + diffusion;

          /*lec->J[peqn][pvar][ii][j] +=  mass + advection + porous + diffusion + source; */
        }
      }

      /*
       * J_m_P
       */

      var = CUDA_PRESSURE;
      pvar = eqn_index[var];

      J = lec->J[peqn][pvar][i];

      for (j = 0; j < eqn_dof[var]; j++) {

        /*  phi_j = bf[var]->phi[j]; */

        diffusion = 0.;

        for (p = 0; p < dim; p++) {
          for (q = 0; q < dim; q++) {

            diffusion -= grad_phi_i_e_a[p][q] * d_Pi->P[q][p][j];

          }
        }
        diffusion *= d_area;

        /*lec->J[peqn][pvar][ii][j] += diffusion ;  */
        J[j] += diffusion;
      }

    } /* end of for(i=eqn_dof*/
  }

  return (status);
}

/* assemble_continuity -- assemble Residual &| Jacobian for continuity eqns
 *
 * in:
 * 	ei -- pointer to Element Indeces	structure
 *	pd -- pointer to Problem Description	structure
 *	af -- pointer to Action Flag		structure
 *	bf -- pointer to Basis Function		structure
 *	fv -- pointer to Field Variable		structure
 *	cr -- pointer to Constitutive Relation	structure
 *	md -- pointer to Mesh Derivative	structure
 *	me -- pointer to Material Entity	structure
 * 	ija -- vector of pointers into the a matrix
 * 	a  -- global Jacobian matrix
 * 	R  -- global residual vector
 *
 * out:
 *	a  -- gets loaded up with proper contribution
 * 	r  -- residual RHS vector
 *
 * Created:	Wed Mar  2 09:27:30 MST 1994 pasacki@sandia.gov
 *
 * Revised:	Sun Mar 20 13:24:50 MST 1994 pasacki@sandia.gov
 */
__host__ __device__ int assemble_continuity_light(
						  int dim, field_variables *fv, basis_function **bf,
						  local_element_contributions *lec, int *eqn_index, int *eqn_dof, double mu)
{
  int p, b;

  int eqn, var;
  int peqn, pvar;

  int i, j;
  int status;

  double div_v = fv->div_v; /* Divergence of v. */

  double advection;

  double det_J;
  double h3;
  double wt;
  double d_area;

  /*
   * Galerkin weighting functions...
   */

  double phi_i;
  double div_phi_j_e_b;
  double *J;

  status = 0;

  //  printf("fv: %g\t%g\t%g\n", fv->v[0], fv->v[1], fv->P);

  /*
   * Unpack variables from structures for local convenience...
   */

  eqn = CUDA_PRESSURE;
  peqn = eqn_index[eqn];

  wt = fv->wt;
  det_J = bf[eqn]->detJ; /* Really, ought to be mesh eqn. */
  h3 = fv->h3; /* Differential volume element (scales). */

  d_area = wt * det_J * h3;

  /*
   * Get the deformation gradients and tensors if needed
   */

  //  if (af->Assemble_Residual)
  //    {
  for (i = 0; i < eqn_dof[eqn]; i++) {
    phi_i = bf[eqn]->phi[i];

    /*
     *  Advection:
     *    This term refers to the standard del dot v .
     *
     *    int (phi_i div_v d_omega)
     *
     *   Note density is not multiplied into this term normally
     */

    advection = div_v;

    advection *= phi_i * d_area;

    /*
     *  Add up the individual contributions and sum them into the local element
     *  contribution for the total continuity equation for the ith local unknown
     */
    lec->R[peqn][i] += advection;
  }
  //    }

  //  if (af->Assemble_Jacobian)
  //    {
  for (i = 0; i < eqn_dof[eqn]; i++) {
    phi_i = bf[eqn]->phi[i];

    /*
     * J_c_v NOTE that this is applied whenever velocity is a variable
     */
    for (b = 0; b < dim; b++) {
      var = CUDA_VELOCITY1 + b;
      pvar = eqn_index[var];

      J = lec->J[peqn][pvar][i];

      for (j = 0; j < eqn_dof[var]; j++) {

        advection = 0.;

        div_phi_j_e_b = 0.;
        for (p = 0; p < dim; p++) {
          div_phi_j_e_b += bf[var]->grad_phi_e[j][b][p][p];
        }

        advection = phi_i * div_phi_j_e_b * d_area;

        J[j] += advection;

      }
    }

    //	}
  }
  return (status);
}

/*
 * Calculate the total stress tensor for a fluid at a single gauss point
 *  This includes the diagonal pressure contribution
 *
 *  Pi = stress tensor
 *  d_Pi = dependence of the stress tensor on the independent variables
 */
__host__ __device__ void fluid_stress_light(int dim, field_variables *fv,
                                            basis_function **bf, int *eqn_index,
                                            int *eqn_dof, double Pi[CUDA_DIM][CUDA_DIM],
                                            stress_dependence *d_Pi, double mu)
{

  /*
   * Variables for vicosity and derivative
   */
  double *grad_v[CUDA_DIM]; /* Gradient of v. */
  double gamma[CUDA_DIM][CUDA_DIM]; /* shrearrate tensor based on velocity */
  double P;

  int a, b, p, q, j, var;

  /*
   * Field variables...
   */

  P = fv->P;

  /*
   * In Cartesian coordinates, this velocity gradient tensor will
   * have components that are...
   *
   * 			grad_v[a][b] = d v_b
   *				       -----
   *				       d x_a
   */

  for (a = 0; a < dim; a++) {
    grad_v[a] = fv->grad_v[a];
  }

  /* load up shear rate tensor based on velocity */
  for (a = 0; a < dim; a++) {
    for (b = 0; b < dim; b++) {
      gamma[a][b] = grad_v[a][b] + grad_v[b][a];
    }
  }

  for (a = 0; a < dim; a++) {
    for (b = 0; b < dim; b++) {
      Pi[a][b] = -P * (double) delta(a, b) + mu * gamma[a][b];
    }

  }

  for (p = 0; p < dim; p++) {
    for (q = 0; q < dim; q++) {
      for (b = 0; b < dim; b++) {
        for (j = 0; j < eqn_dof[CUDA_VELOCITY1]; j++) {
          /* grad_phi_e cannot be the same for all
           * velocities for 3d stab of 2d flow!!
           * Compare with the old way in the CYLINDRICAL
           * chunk below... */
          d_Pi->v[p][q][b][j] = mu
	    * (bf[CUDA_VELOCITY1 + q]->grad_phi_e[j][b][p][q]
	       + bf[CUDA_VELOCITY1 + p]->grad_phi_e[j][b][q][p]);
        }
      }
    }
  }

  var = CUDA_PRESSURE;
  for (p = 0; p < dim; p++) {
    for (q = 0; q < dim; q++) {
      for (j = 0; j < eqn_dof[var]; j++) {
        d_Pi->P[p][q][j] = -(double) delta(p, q) * bf[var]->phi[j];
      }
    }

  }
}


__device__ void setup_basis_functions(basis_function **bfs)
{

}



__global__ void matrix_fill_kernel(mesh_data *mesh,
				   basis_function **bfs,
				   element *elements,
				   problem_data *mat_system,
				   int num_elements)
{

  int tid = threadIdx.x + blockIdx.x * blockDim.x;


  if (tid < num_elements) {
    double mu = 1;
    double rho = 1;
    int ielem = tid;
    field_variables fv_st;
    field_variables *fv = &fv_st;
    local_element_contributions lec_st;
    local_element_contributions *lec = &lec_st;

    basis_function *bfe[3];
    basis_function q2_bf_st;
    basis_function *q2_bf = &q2_bf_st;
    basis_function q1_bf_st;
    basis_function *q1_bf = &q1_bf_st;

    q2_bf->interpolation = BIQUAD_QUAD;
    q1_bf->interpolation = BILINEAR_QUAD;

    q2_bf->element_shape = BIQUAD_QUAD;
    q1_bf->element_shape = BIQUAD_QUAD;

    q2_bf->shape = BIQUAD_QUAD;
    q1_bf->shape = BILINEAR_QUAD;

    q2_bf->num_dof = 9;
    q1_bf->num_dof = 4;

    bfe[CUDA_VELOCITY1] = q2_bf;
    bfe[CUDA_VELOCITY2] = q2_bf;
    bfe[CUDA_PRESSURE] = q1_bf;

    for (int e = 0; e < CUDA_NUM_EQNS; e++) {
      elements[ielem].eqn_index[e] = e;
      elements[ielem].eqn_dof[e] = bfe[e]->num_dof;
    }

    //    element_stiffness_pointers *esp = &elements[ielem].esp;

    zero_lec(lec);
    //    load_esp(esp, elements[ielem], bfe, mat_system, mesh);

    int ip = 0;
    for (ip = 0; ip < CUDA_MDE; ip++) {

      fv->wt = find_gauss_weight(bfe[CUDA_VELOCITY1], ip);
      find_gauss_points(fv->xi, bfe[CUDA_VELOCITY1], ip);
      for (int eqn = 0; eqn < CUDA_NUM_EQNS; eqn++) {
	load_basis_functions(mesh->dim, fv->xi, bfe[eqn]);
      }

      load_fv(mat_system, fv, bfe, &(elements[ielem]));

      jacobian_inverse_and_mapping(mesh, fv, bfe, elements[ielem]);

      for (int eqn = 0; eqn < CUDA_NUM_EQNS; eqn++) {
	load_basis_functions_grad(fv, bfe[eqn]);
      }

      load_fv_grad(mat_system, fv, bfe, &(elements[ielem]));

      assemble_momentum_light(CUDA_DIM, fv, bfe, lec,  elements[ielem].eqn_index,  elements[ielem].eqn_dof, 0, 0,
			      0, mu, rho);

      assemble_continuity_light(CUDA_DIM, fv, bfe, lec,  elements[ielem].eqn_index,  elements[ielem].eqn_dof,
				mu);

    }
    set_dirichlet(mesh, lec, elements[ielem]);
    load_lec_gpu(lec, mesh, bfe, elements[ielem], mat_system);
  }
}



void matrix_fill_gpu(mesh_data *mesh, basis_function **bfs, element *elements,
		     problem_data *mat_system) {

  cudaDeviceSynchronize();
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  int threads_per_block = 256;
  int blocks_per_grid = (mesh->num_elem + threads_per_block - 1) / threads_per_block;
  matrix_fill_kernel<<<blocks_per_grid, threads_per_block>>>(mesh, bfs, elements, mat_system, mesh->num_elem);
  cudaDeviceSynchronize();

  end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end-start;

  std::cout << "gpu asm: " << elapsed_seconds.count();
  /*
  start = std::chrono::system_clock::now();

  for (int ielem = 0; ielem < mesh->num_elem; ielem++) {
    load_lec(&(elements[ielem].lec), mesh, elements[ielem].bfs, elements[ielem], mat_system);
  }

  end = std::chrono::system_clock::now();

  elapsed_seconds = end-start;

  std::cout << " load: " << elapsed_seconds.count();
  */
}


void solve_umfpack(problem_data *mat_system) {
  int n = mat_system->A->m;
  int *Ap, *Ai;
  double *Ax;


  Ap = mat_system->A->rowptr;
  Ai = mat_system->A->colind;
  Ax = mat_system->A->val;

  double *null = (double *) NULL ;

  double *b = mat_system->b;

  double *x = mat_system->delta_x;

  void *Symbolic, *Numeric ;
  (void) umfpack_di_symbolic (n, n, Ap, Ai, Ax, &Symbolic, null, null) ;
  (void) umfpack_di_numeric (Ap, Ai, Ax, Symbolic, &Numeric, null, null) ;
  umfpack_di_free_symbolic (&Symbolic) ;
  (void) umfpack_di_solve (UMFPACK_At, Ap, Ai, Ax, x, b, Numeric, null, null) ;
  umfpack_di_free_numeric (&Numeric) ;

  int *indices = new int[n];
  for (int i = 0; i < n; i++) {
    indices[i] = i;
  }
}


void zero_vec(int n, double *vec)
{
  for (int i = 0; i < n; i++) {
    vec[i] = 0.0;
  }
}

double l2_norm(int n, double *vec)
{
  double norm = 0;
  for (int i = 0; i < n; i++) {
    norm += vec[i] * vec[i];
  }
  return sqrt(norm);
}


int cuda_solve_nonlinear(int num_newton_iterations, double residual_tolerance,
		    basis_function** bfs, element *elements, problem_data *mat_system,
		    mesh_data *mesh, const Epetra_Comm & comm) {
  int converged = 0;
  int newt_it = 0;
  double residual_L2_norm = 0;
  double update_L2_norm;

  if (comm.MyPID() == 0) {
    std::cout << "It\t" << "Resid L2\t" << "Update L2" << std::endl;
  }

  std::chrono::time_point<std::chrono::system_clock> start, end;

  while (!converged && newt_it < num_newton_iterations) {

    if (comm.MyPID() == 0) {
      zero_vec(mat_system->A->m, mat_system->delta_x);
      zero_vec(mat_system->A->m, mat_system->b);
      zero_vec(mat_system->A->nnz, mat_system->A->val);
      /* Matrix Fill */
      matrix_fill_gpu(mesh, bfs, elements, mat_system);

      residual_L2_norm = l2_norm(mat_system->A->m, mat_system->b);
      /* row sum scaling */
      row_sum_scale(mat_system);
      /* Matrix solve */
      //qr_solve(mat_system);


      start = std::chrono::system_clock::now();

      //      solve_umfpack(mat_system);
    }

    cuda_trilinos_solve(comm, mat_system);

    if (comm.MyPID() == 0) {
      end = std::chrono::system_clock::now();

      std::chrono::duration<double> elapsed_seconds = end-start;

      std::cout << " slv: " << elapsed_seconds.count() << std::endl;

      /* compute l2 norm */

      update_L2_norm = l2_norm(mat_system->A->m, mat_system->delta_x);

      /* check converged */
      converged = residual_L2_norm < residual_tolerance;
      std::cout << newt_it << "\t" << residual_L2_norm << "\t"
		<< update_L2_norm << std::endl;

      for (int i = 0; i < mat_system->A->m; i++) {
	mat_system->x[i] -= mat_system->delta_x[i];
      }
    }

    comm.Broadcast(&converged, 1, 0);
    newt_it++;
  }
  return converged;
}

}
