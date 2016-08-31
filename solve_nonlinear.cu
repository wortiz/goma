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
#include "ns_macros.h"
#include "cublas_v2.h"
#include "cusparse_v2.h"

extern "C" {
#include "exo_read_mesh.h"
}
#include "ns_structs.h"
#include "solve_nonlinear.cuh"
//#include "ns_basis_functions.cuh"
//#include "ns_assemble.cuh"
#include <iostream>
#define checkCudaErrors(val) val
#include "umfpack.h"
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/print.h>
#include <cusp/krylov/bicgstab.h>

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

__host__ __device__ int index_solution(matrix_data *mat_system, int node, int eqn) {
  int e = mat_system->eqn_index[eqn];
  if (e == -1) return -1;
  return mat_system->solution_index[e][node];
}

__host__ __device__ void row_sum_scale(matrix_data *mat_system) {
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

__host__ __device__ void load_fv(matrix_data **mat_systems, field_variables *fv,
				 field_variables *fv_old,
				 basis_function **bfs, element *elem) {

  fv->h[0] = 1;
  fv->h[1] = 1;
  fv->h3 = 1;

  int v, a;
  int idx;
  for (v = AUX_VELOCITY1, a = 0; v <= AUX_VELOCITY2; v++, a++) {
    fv->v_star[a] = 0.0;
    fv_old->v_star[a] = 0.0;

    for (int i = 0; i < bfs[v]->num_dof; i++) {
      idx = index_solution(mat_systems[0], elem->gnn[i] - 1, v);
      fv->v_star[a] += mat_systems[0]->x[idx] * bfs[v]->phi[i];
      fv_old->v_star[a] += mat_systems[0]->x_old[idx] * bfs[v]->phi[i];
    }
  }

  fv->P_star = 0;
  fv_old->P_star = 0;

  v = AUX_PRESSURE;
  for (int i = 0; i < bfs[v]->num_dof; i++) {
    idx = index_solution(mat_systems[1], elem->gnn[i] - 1, v);
    fv->P_star += mat_systems[1]->x[idx] * bfs[v]->phi[i];
    fv_old->P_star += mat_systems[1]->x_old[idx] * bfs[v]->phi[i];
  }

  for (v = VELOCITY1, a = 0; v <= VELOCITY2; v++, a++) {
    fv->v[a] = 0.0;
    fv_old->v[a] = 0.0;
    for (int i = 0; i < bfs[v]->num_dof; i++) {
      idx = index_solution(mat_systems[2], elem->gnn[i] - 1, v);
      fv->v[a] += mat_systems[2]->x[idx] * bfs[v]->phi[i];
      fv_old->v[a] += mat_systems[2]->x_old[idx] * bfs[v]->phi[i];
    }
  }

  fv->P = 0;
  fv_old->P = 0;

  v = PRESSURE;
  for (int i = 0; i < bfs[PRESSURE]->num_dof; i++) {
    idx = index_solution(mat_systems[3], elem->gnn[i] - 1, v);
    fv->P += mat_systems[3]->x[idx] * bfs[PRESSURE]->phi[i];
    fv_old->P += mat_systems[3]->x_old[idx] * bfs[PRESSURE]->phi[i];
  }
}

__host__ __device__ void load_fv_grad(matrix_data **mat_systems, field_variables *fv,
				      field_variables *fv_old,
				      basis_function **bfs, element *elem) {
  int p, q, r, i;
  int v;
  int dofs;


  v = AUX_VELOCITY1;
  dofs = bfs[v]->num_dof;
  for (p = 0; p < DIM; p++) {
    for (q = 0; q < DIM; q++) {
      fv->grad_v_star[p][q] = 0.0;
      fv_old->grad_v_star[p][q] = 0.0;
            for (r = 0; r < DIM; r++) {
	for (i = 0; i < dofs; i++) {
	  int idx = index_solution(mat_systems[0], elem->gnn[i] - 1, v+r);
	  fv->grad_v_star[p][q] += mat_systems[0]->x[idx]
	    * bfs[v]->grad_phi_e[i][r][p][q];
	  fv_old->grad_v_star[p][q] += mat_systems[0]->x_old[idx]
	    * bfs[v]->grad_phi_e[i][r][p][q];

	}
      }
    }
  }

  fv->div_v_star = fv->grad_v_star[0][0] + fv->grad_v_star[1][1];
  fv_old->div_v_star = fv_old->grad_v_star[0][0] + fv_old->grad_v_star[1][1];

  v = AUX_PRESSURE;
  dofs = bfs[v]->num_dof;
  for (p = 0; p < DIM; p++) {
    fv->grad_P_star[p] = 0.0;
    fv_old->grad_P_star[p] = 0.0;

    for (i = 0; i < dofs; i++) {
      int idx = index_solution(mat_systems[1], elem->gnn[i] - 1, v);
      fv->grad_P_star[p] += mat_systems[1]->x[idx] * bfs[v]->grad_phi[i][p];
      fv_old->grad_P_star[p] += mat_systems[1]->x_old[idx] * bfs[v]->grad_phi[i][p];
    }
  }

  v = VELOCITY1;
  dofs = bfs[v]->num_dof;
  for (p = 0; p < DIM; p++) {
    for (q = 0; q < DIM; q++) {
      fv->grad_v[p][q] = 0.0;
      fv_old->grad_v[p][q] = 0.0;
      for (r = 0; r < DIM; r++) {
	for (i = 0; i < dofs; i++) {
	  int idx = index_solution(mat_systems[2], elem->gnn[i] - 1, v+r);
	  fv->grad_v[p][q] += mat_systems[2]->x[idx]
	    * bfs[v]->grad_phi_e[i][r][p][q];
	  fv_old->grad_v[p][q] += mat_systems[2]->x_old[idx]
	    * bfs[v]->grad_phi_e[i][r][p][q];

	}
      }
    }
  }

  fv->div_v = fv->grad_v[0][0] + fv->grad_v[1][1];
  fv_old->div_v = fv_old->grad_v[0][0] + fv_old->grad_v[1][1];

  v = PRESSURE;
  dofs = bfs[v]->num_dof;
  for (p = 0; p < DIM; p++) {
    fv->grad_P[p] = 0.0;
    fv_old->grad_P[p] = 0.0;

    for (i = 0; i < dofs; i++) {
      int idx = index_solution(mat_systems[3], elem->gnn[i] - 1, v);
      fv->grad_P[p] += mat_systems[3]->x[idx] * bfs[v]->grad_phi[i][p];
      fv_old->grad_P[p] += mat_systems[3]->x_old[idx] * bfs[v]->grad_phi[i][p];
    }
  }

}

__host__ __device__ void set_dirichlet(mesh_data *mesh, local_element_contributions *lec,
				       element elem, int num_eqns, int *eqn_index) {
  for (int i = 0; i < elem.num_bcs; i++) {
    int eqn = elem.bcs[i].eqn;
    int e = eqn_index[eqn];
    if (e != -1) {
      for (int j = 0; j < MDE; j++) {
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

	    for (b = 0; b < num_eqns; b++) {
	      for (d = 0; d < MDE; d++) {
		lec->J[e][b][j][d] = 0.0;
	      }

	    }
	    lec->J[e][e][j][j] = 1;

	    lec->R[e][j] = 0.0;
	  }
	}
      }
    }
  }
}

__device__ void load_lec_gpu(local_element_contributions *lec, mesh_data *mesh,
				  basis_function **bfs, element elem, matrix_data *mat_system) {

  for (int e = 0; e < mat_system->num_eqns; e++) {
    int eqn = mat_system->inv_eqn_index[e];
    assert(eqn != -1);
    for (int i = 0; i < bfs[eqn]->num_dof; i++) {
      int row = index_solution(mat_system, elem.gnn[i] - 1, eqn);
      assert(row != -1);
      atomicAdd_double(&(mat_system->b[row]), lec->R[e][i]);
      for (int v = 0; v < mat_system->num_eqns; v++) {
	int var = mat_system->inv_eqn_index[v];
	assert(var != -1);
	for (int j = 0; j < bfs[var]->num_dof; j++) {
	  int col = index_solution(mat_system, elem.gnn[j] - 1, var);
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

void load_lec_cpu(local_element_contributions *lec, mesh_data *mesh,
		  basis_function **bfs, element elem, matrix_data *mat_system) {

#if 0
  static FILE *fp = NULL;

  if (fp == NULL)
    fp = fopen("jac_cu.txt", "w");

  static int call = 0;

  call++;

  if (call <= 4) {
    for (int e = 0; e < 2; e++) {
      for (int v = 0; v < 2; v++) {
	for (int i = 0; i < 9; i++) {
	  for (int j = 0; j < 9; j++) {
	    fprintf(fp, "%d: %d %d %d %d %g\n", call, e, v, i, j, lec->J[e][v][i][j]);
	  }
	}
      }
    }
  } else {
    fclose(fp);
    exit(0);
  }
#endif
  for (int e = 0; e < mat_system->num_eqns; e++) {
    int eqn = mat_system->inv_eqn_index[e];
    assert(eqn != -1);
    for (int i = 0; i < bfs[eqn]->num_dof; i++) {
      int row = index_solution(mat_system, elem.gnn[i] - 1, eqn);
      assert(row != -1);
      mat_system->b[row] += lec->R[e][i];
      for (int v = 0; v < mat_system->num_eqns; v++) {
	int var = mat_system->inv_eqn_index[v];
	assert(var != -1);
	for (int j = 0; j < bfs[var]->num_dof; j++) {
	  int col = index_solution(mat_system, elem.gnn[j] - 1, var);
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

__host__ __device__ void zero_lec(local_element_contributions *lec)
{
  for (int e = 0; e < 2; e++) {
    for (int i = 0; i < MDE; i++) {
      lec->R[e][i] = 0;
    }
    
    for (int v = 0; v < 2; v++) {
      for (int i = 0; i < MDE; i++) {
	for (int j = 0; j < MDE; j++) {
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

  basis_function *shape_bf = bfs[VELOCITY1];

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

  for (int i = 0; i < NUM_EQNS; i++) {
    if (i == VELOCITY1)
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
    for (p = 0; p < DIM; p++) {
      bfv->grad_phi[i][p] = (bfv->d_phi[i][p]) / (fv->h[p]);
    }
  }

  for (i = 0; i < dofs; i++) {
    for (p = 0; p < DIM; p++) {
      for (a = 0; a < DIM; a++) {
        for (q = 0; q < DIM; q++) {
          if (q == a)
            bfv->grad_phi_e[i][a][p][a] = bfv->grad_phi[i][p];
          else
            bfv->grad_phi_e[i][a][p][q] = 0.0;
        }
      }
    }
  }
}

/*
 *This function assembles the first step of the CBS, split-B, quasi-implicit
 * method.  Here, we solve for an auxiliary velocity from some form of the 
 * momentum equation.  Due to the time discretization, the only non-linear term
 * is mu_star*grad_v_star if mu_star depends on v_star.
 */
__host__ __device__ int
assemble_aux_u(int dim,
	       field_variables *fv,
	       field_variables *fv_old,
	       basis_function **bf,
	       local_element_contributions *lec,
	       int *eqn_index,
	       double time,   // Current time
	       double dt,
	       double rho,
	       double mu)     // Current time step
{
  int eqn, peqn, var, pvar, status=0;
  int a, b, i, ii, j, p, q;

  // Relevant field variable quantities
  double *v_star, *v_old, *grad_v_star[DIM], *grad_v_old[DIM], P_old;
  double gamma_star[DIM][DIM], gamma_old[DIM][DIM];

  // Residual contributions
  double advection, diffusion, diffusion_a, mass;
  
  // Residual and Jacobian
  double *R, *J=NULL;
 
  // Integration factors
  double h3, wt, det_J;
  
  // Basis functions
  basis_function *bfm;
  double phi_i, *phi_i_vector, (*grad_phi_i_e_a)[DIM]=NULL;
  double phi_j, *phi_j_vector;
  
  
  // Set equation index
  //eqn = R_MOMENTUM1;
  eqn = VELOCITY1;
  
  // Fill field variables and parameters
  wt = fv->wt;
  det_J = bf[eqn]->detJ;	       
  h3 = fv->h3;
  
  //v_star = fv->v_star;
  v_star = fv->v_star;
  v_old = fv_old->v;
  //v_old = (pg->sbcfv).v_old;
  //P_old = (pg->sbcfv).P_old;
  P_old = fv_old->P;
  

  for(a=0; a<dim; a++)
    {
      grad_v_star[a] = fv->grad_v_star[a];
      //grad_v_star[a] = fv->grad_v[a];
      //grad_v_old[a] = (pg->sbcfv).grad_v_old[a];
      grad_v_old[a] = fv_old->grad_v[a];
    }

  for(a=0; a<dim; a++)
    {
      for(b=0; b<dim; b++)
	{
	  gamma_star[a][b] = grad_v_star[a][b] + grad_v_star[b][a];
	  gamma_old[a][b] = grad_v_old[a][b] + grad_v_old[b][a];
	}
    }

  // Start assembling that residual

  for(a=0; a<dim; a++)
    {
      eqn = AUX_VELOCITY1 + a;
      //eqn  = R_MOMENTUM1 + a;
      bfm  = bf[eqn];
      phi_i_vector = bfm->phi;

      peqn = eqn_index[eqn];
      R = lec->R[peqn];


      for(i=0; i<bf[eqn]->num_dof; i++) 
	{	    
	  ii = i;
		  
	  phi_i = phi_i_vector[i];
	  grad_phi_i_e_a = bfm->grad_phi_e[i][a];
		  
	  /*
	   * Time derivative contribution, typically fv_dot is used
	   * However, our time discretization does not fit that of the
	   * backward Euler or Crank-Nicolson schemes, so we need to do 
	   * this separately
	   */
	  mass = 0.0;
	  mass += rho/dt*(v_star[a]-v_old[a]);
	  mass *= phi_i*h3*wt*det_J;
		  
	  // Advection contribution, uses old velocity only
	  advection = 0.0;
	  for(p=0; p<dim; p++)
	    {
	      advection += v_old[p]*grad_v_old[p][a];
	    }
	  advection *= rho*phi_i*h3*wt*det_J;
		  
	  // Diffusion contribution, uses both old and aux velocity
	  diffusion = 0.0;
	  for(p=0; p<dim; p++) 
	    {
	      for(q=0; q<dim; q++) 
		{
		  diffusion += mu/2.0*gamma_star[q][p]*grad_phi_i_e_a[p][q];
		  diffusion += mu/2.0*gamma_old[q][p]*grad_phi_i_e_a[p][q];
			      
		  if(p==q)
		    {
		      diffusion -= P_old*grad_phi_i_e_a[p][q];
		    }
		}
	    }		      
	  diffusion *= h3*wt*det_J;
		  
	  R[ii] += mass + advection + diffusion;
	}     // for i<dof[eqn]
    }         // for a<wim

  

  for(a=0; a<dim; a++)
    {
      eqn = AUX_VELOCITY1 + a;
      //eqn  = R_MOMENTUM1 + a;
      peqn = eqn_index[eqn];
      bfm  = bf[eqn];	 
	  
      phi_i_vector = bfm->phi;
	  
      for(i=0; i<bf[eqn]->num_dof; i++) 
	{		  
	  ii = i;
	  phi_i = phi_i_vector[i];		  
	  grad_phi_i_e_a = bfm->grad_phi_e[i][a];	
		  
	  // J_v_star
	  for(b=0; b<dim; b++)
	    {
	      var = AUX_VELOCITY1 + b;
	      //var = VELOCITY1 + b;
	      pvar = eqn_index[var];
	      J = lec->J[peqn][pvar][ii];			  
	      phi_j_vector = bf[var]->phi;
			  
	      for(j=0; j<bf[var]->num_dof; j++)
		{			      
		  phi_j = phi_j_vector[j];			      

		  mass = 0.0;
		  if (a == b) {
		    mass += rho/dt*phi_j;
		    mass *= phi_i*h3*wt*det_J;
		  }
			      
		  diffusion = 0.0;

		  for(p=0; p<dim; p++)
		    {
		      for(q=0; q<dim; q++)
			{
			  diffusion_a = 0.0;
			  diffusion_a += bf[AUX_VELOCITY1+p]->grad_phi_e[j][b][q][p];
			  diffusion_a += bf[AUX_VELOCITY1+q]->grad_phi_e[j][b][p][q];
			  //diffusion_a += bf[VELOCITY1+p]->grad_phi_e[j][b][q][p];
			  //diffusion_a += bf[VELOCITY1+q]->grad_phi_e[j][b][p][q];

			  diffusion += mu/2.0*diffusion_a*grad_phi_i_e_a[p][q];
			  //diffusion += d_mu->v[b][j]*gamma_star[p][q]*grad_phi_i_e_a[p][q];
			}
		    }
		  diffusion *= h3*wt*det_J;
			      
		  J[j] +=  mass + diffusion;
		} 
	    }	      	  
	}     // for i<dof[eqn]
    }         // for a<wim

  return(status);
} // assemble_aux_u
  

/*
 * This function assembles the pressure-poisson equation for the CBS, split-B, quasi-implicit method
 * Here, we solve for an intermediate/correction pressure P_star
 * It is assumed [grad(P_star)-v_star] dot n = 0 on the boundary which allows for the boundary conditions
 * from the auxiliary velocity step to translate directly through the algorithm
 */
__host__ __device__ int
assemble_press_poisson(int dim,
		       field_variables *fv,
		       field_variables *fv_old,
		       basis_function **bf,
		       local_element_contributions *lec,
		       int *eqn_index,
		       double time,   // Current time
		       double dt,
		       double rho,
		       double mu)
{
  // Some indices
  int a, i, j, eqn, peqn, var, pvar, status=0;

  // Relevant field variable quantities
  double *v_star, *grad_P_star;

  // Residual contributions
  double diffusion, mass;

  // Residual and Jacobian
  double *R, *J;
 
  // Integration factors
  double h3, wt, det_J;
  
  // Basis functions
  double (*grad_phi)[DIM];
  
  // Set equation index
  eqn = AUX_PRESSURE;
  peqn = eqn_index[eqn];


  R = lec->R[peqn];
  wt = fv->wt;
  det_J = bf[eqn]->detJ;	       
  h3 = fv->h3;
  grad_phi = bf[eqn]->grad_phi;

  //  div_v_star = fv->div_v_star;
  //grad_P_star = fv->grad_P_star;
  //div_v_star = (pg->sbcfv).div_v_star;
  //v_star = (pg->sbcfv).v_star;
  v_star = fv->v_star;
  grad_P_star = fv->grad_P_star;
  //  grad_P_old = fv_old->grad_P;

  // Residual

  for(i=0; i<bf[eqn]->num_dof; i++)
    {
	  
      // Incompressibility correction
      mass = 0.0;
      for(a=0; a<dim; a++)
	{
	  mass -= v_star[a]*grad_phi[i][a];
	}
	      
      //mass += div_v_star*phi_i;
      mass *= rho/dt*h3*wt*det_J;

      // Pressure laplacian contribution
      diffusion = 0.0;
      for(a=0; a<dim; a++)
	{
	  diffusion += grad_P_star[a]*grad_phi[i][a];
	}
      diffusion *= h3*wt*det_J;

      R[i] += mass + diffusion;
	  
    } // for i<dof[eqn]


  // Jacobian
  for(i=0; i<bf[eqn]->num_dof; i++)
    { 
      // J_P_star
      //var = AUX_PRESSURE
      var = AUX_PRESSURE;
      pvar = eqn_index[var];
      J = lec->J[peqn][pvar][i];	      
      for(j=0; j<bf[var]->num_dof; j++)
	{		  
	  diffusion = 0.0;
	  for(a=0; a<dim; a++)
	    {
	      diffusion += grad_phi[i][a]*bf[var]->grad_phi[j][a];
	    }		      
	  diffusion *= h3*wt*det_J;
	  J[j] += diffusion;
	}	  	
  
    } // for i<dof[eqn] 

  return(status);
} // assemble_poisson_press



/*
 * This function assembles the pressure projection step of the CBS, split-B, quasi-implicit method
 * Here, we obtain the velocity from the pressure projection of the auxiliary velocity
 */

__host__ __device__ int assemble_press_proj(int dim,
					    field_variables *fv,
					    field_variables *fv_old,
					    basis_function **bf,
					    local_element_contributions *lec,
					    int *eqn_index,
					    double time,   // Current time
					    double dt,
					    double rho,
					    double mu)
{
  // Some indices
  int eqn, peqn, var, pvar, status=0;
  int a, b, i, j;

  // Relevant field variable quantities
  double *v_star, *v, *grad_P_star;

  // Residual contributions
  double diffusion, mass;
  
  // Residual and Jacobian
  double *R, *J;
 
  // Integration factors
  double h3, wt, det_J;
  
  // Basis functions
  basis_function *bfm;
  double phi_i, *phi_i_vector;
  double phi_j, *phi_j_vector;
  
  // Set equation index
  //eqn = R_PRESSURE_PROJECTION1
  eqn = VELOCITY1;

  wt = fv->wt;
  det_J = bf[eqn]->detJ;	       
  h3 = fv->h3;

  v_star = fv->v_star;
  //v_star = (pg->sbcfv).v_star;
  v = fv->v;
  grad_P_star = fv->grad_P_star;
    //  P_star = (pg->sbcfv).P_star;
  //grad_P_star = (pg->sbcfv).grad_P_star;

  // Start assembling that residual
  for(a=0; a<dim; a++)
    {
      //eqn = R_PRESSURE_PROJECTION1 + a;
      eqn  = VELOCITY1 + a;
      peqn = eqn_index[eqn];
      bfm  = bf[eqn];

      R = lec->R[peqn];
      phi_i_vector = bfm->phi;

      for(i=0; i<bf[eqn]->num_dof; i++) 
	{	    
	  phi_i = phi_i_vector[i];
		  
	  // Velocity contribution
	  mass = 0.0;
	  mass += v[a] - v_star[a];
	  mass *= rho/dt*phi_i*h3*wt*det_J;
		  
	  // Pressure contribution
	  diffusion = 0.0;
	  //for(b=0; b<dim; b++)
	  //{
	  //  diffusion -= P_star*grad_phi_i_e_a[b][b];
	  //}

	  diffusion += grad_P_star[a]*phi_i;
	  diffusion *= h3*wt*det_J;
		  
	  R[i] += mass + diffusion;		  

	}     // for i<dof[eqn]
    }         // for a<dim

  // Jacobian
  for(a=0; a<dim; a++)
    {
      eqn  = VELOCITY1 + a;
      peqn = eqn_index[eqn];
      bfm  = bf[eqn];	 
	  
      phi_i_vector = bfm->phi;
	  
      for(i=0; i<bf[eqn]->num_dof; i++) 
	{		  

	  phi_i = phi_i_vector[i];	
		  
	  // J_v
	  for(b=0; b<dim; b++)
	    {
	      var = VELOCITY1 + b;
	      pvar = eqn_index[var];
	      J = lec->J[peqn][pvar][i];			  
	      phi_j_vector = bf[var]->phi;
			  
	      for(j=0; j<bf[var]->num_dof; j++)
		{			      
		  phi_j = phi_j_vector[j];
			      
		  mass = 0.0;			  			    
		  if(a == b)
		    {
		      mass += phi_i*phi_j;
		      mass *= rho/dt*h3*wt*det_J;
		    }
		  J[j] += mass;
		}
	    }


	}     // for i<dof[eqn]
    }         // for a<dim
  
  return(status);
} // assemble_press_proj

/*
 * This function assembles the pressure update in the CBS, split-B, quasi-implicit method
 * Here, we find the new pressure from the old pressure, auxiliary pressure, and
 * incompressibility
 */
__host__ __device__ int
assemble_press_update(int dim,
		      field_variables *fv,
		      field_variables *fv_old,
		      basis_function **bf,
		      local_element_contributions *lec,
		      int *eqn_index,
		      double time,   // Current time
		      double dt,
		      double rho,
		      double mu)
{
  // Some indices
  int a, b, i, j, eqn, peqn, var, pvar, status=0;

  // Viscosity and density
  double mu_star;
  
  // Relevant field variable quantities
  double div_v_star, P, P_star, P_old, *grad_P_old;
  double *grad_v_star[DIM], gamma_star[DIM][DIM];

  // Residual contributions
  double diffusion, mass;

  // Residual
  double *R, *J;
 
  // Integration factors
  double h3, wt, det_J;
  
  // Basis functions
  double phi_i, phi_j, (*grad_phi)[DIM];
  
  // Set equation index
  //eqn = R_PRESSURE_UPDATE;
  eqn = PRESSURE;
  peqn = eqn_index[eqn];
  R = lec->R[peqn];
  wt = fv->wt;
  det_J = bf[eqn]->detJ;	       
  h3 = fv->h3;
  grad_phi = bf[eqn]->grad_phi;

  div_v_star = fv->div_v_star;
  //grad_v_star = fv->grad_v_star;
  //div_v_star = (pg->sbcfv).div_v_star;
  P = fv->P;
  P_star = fv->P_star;
  //P_star = (pg->sbcfv).P_star;
  P_old = fv_old->P;
  grad_P_old = fv_old->grad_P;

  for(a=0; a<dim; a++)
    {
      grad_v_star[a] = fv->grad_v_star[a];
      //grad_v_star[a] = (pg->sbcfv).grad_v_star[a];
    }
  
  mu_star = mu;


  // Residual
  for(i=0; i<bf[eqn]->num_dof; i++)
    {
      phi_i = bf[eqn]->phi[i];
	  
      // Pressure contribution
      mass = 0.0;

      mass += P - P_star - P_old;
      mass *= phi_i*h3*wt*det_J;

      // Incompressibility contribution
      diffusion = 0.0;
      diffusion += mu_star/2.0*div_v_star*phi_i;
      /*	      
		      for(a=0; a<dim; a++)
		      {
		      diffusion -= mu_star*dt*grad_P_old[a]*grad_phi[i][a];
		      }
      */
      diffusion *= h3*wt*det_J;

      R[i] += mass + diffusion;	  
    } // for i<dof[eqn]

  // Jacobian
  for(i=0; i<bf[eqn]->num_dof; i++)
    { 
      phi_i = bf[eqn]->phi[i];

      // J_P
      var = PRESSURE;
      pvar = eqn_index[var];
      J = lec->J[peqn][pvar][i];	      
      for(j=0; j<bf[var]->num_dof; j++)
	{
	  phi_j = bf[var]->phi[j];

	  mass = 0.0;
	  mass += phi_i*phi_j;		      
	  mass *= h3*wt*det_J;

	  J[j] += mass;
	}
	  
    } // for i<dof[eqn] 

  return(status);
} // assemble_press_update

__global__ void matrix_fill_kernel(double time, double dt, problem_data *pd,
				   int num_elements,
				   int matrix)
{

  int tid = threadIdx.x + blockIdx.x * blockDim.x; 

  if (tid < num_elements) {
    double mu = 0.1;
    double rho = 1;
    int ielem = tid;
    field_variables *fv = &(pd->fvs[tid]);
    field_variables *fv_old = &(pd->fv_olds[tid]);

    local_element_contributions lec_st;
    local_element_contributions *lec = &lec_st;

    basis_function *bfe[NUM_EQNS];

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

    bfe[VELOCITY1] = q2_bf;
    bfe[VELOCITY2] = q2_bf;
    bfe[PRESSURE] = q1_bf;
    bfe[AUX_VELOCITY1] = q2_bf;
    bfe[AUX_VELOCITY2] = q2_bf;
    bfe[AUX_PRESSURE] = q1_bf;

    for (int e = 0; e < NUM_EQNS; e++) {
      pd->elements[ielem].eqn_index[e] = e;
      pd->elements[ielem].eqn_dof[e] = bfe[e]->num_dof;
    }

    //    element_stiffness_pointers *esp = &elements[ielem].esp;

    zero_lec(lec);
    //    load_esp(esp, elements[ielem], bfe, mat_system, pd->mesh);

    int ip = 0;
    for (ip = 0; ip < MDE; ip++) {

      
      fv->wt = find_gauss_weight(bfe[VELOCITY1], ip);
      find_gauss_points(fv->xi, bfe[VELOCITY1], ip);
      for (int eqn = 0; eqn < NUM_EQNS; eqn++) {
	load_basis_functions(pd->mesh->dim, fv->xi, bfe[eqn]);
      }


      load_fv(pd->systems, fv, fv_old, bfe, &(pd->elements[ielem]));

      jacobian_inverse_and_mapping(pd->mesh, fv, bfe, pd->elements[ielem]);

      for (int eqn = 0; eqn < NUM_EQNS; eqn++) {
	load_basis_functions_grad(fv, bfe[eqn]);
      }

      load_fv_grad(pd->systems, fv, fv_old, bfe, &(pd->elements[ielem]));

      switch (matrix) {
      case 0:
	assemble_aux_u(pd->mesh->dim, fv, fv_old, bfe, lec,
		       pd->systems[matrix]->eqn_index,
		       time, dt, rho, mu);
	break;
      case 1:
	assemble_press_poisson(pd->mesh->dim, fv, fv_old, bfe, lec,
		       pd->systems[matrix]->eqn_index,
		       time, dt, rho, mu);
	break;
      case 2:
	assemble_press_proj(pd->mesh->dim, fv, fv_old, bfe, lec,
			    pd->systems[matrix]->eqn_index,
			    time, dt, rho, mu);
	break;
      case 3:
	assemble_press_update(pd->mesh->dim, fv, fv_old, bfe, lec,
			      pd->systems[matrix]->eqn_index,
			      time, dt, rho, mu);
	break;

      default:	
	break;
      }

    }

    set_dirichlet(pd->mesh, lec, pd->elements[ielem], pd->systems[matrix]->num_eqns, pd->systems[matrix]->eqn_index);
    load_lec_gpu(lec, pd->mesh, bfe, pd->elements[ielem], pd->systems[matrix]);

  }
}



void matrix_fill_gpu(double time, double dt, problem_data *pd, int matrix) {

  cudaDeviceSynchronize();  
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  int threads_per_block = 256;
  int blocks_per_grid = (pd->mesh->num_elem + threads_per_block - 1) / threads_per_block;
  matrix_fill_kernel<<<blocks_per_grid, threads_per_block>>>(time, dt, pd, pd->num_elements, matrix);
  cudaDeviceSynchronize();

  end = std::chrono::system_clock::now();
 
  std::chrono::duration<double> elapsed_seconds = end-start;
 
  std::cout << "gpu asm: " << elapsed_seconds.count();
}

void matrix_fill_single(problem_data *pd,
			int tid,
			int matrix)
{

  double mu = 0.1;
  double rho = 1;
  int ielem = tid;
  field_variables *fv = &(pd->fvs[tid]);
  field_variables *fv_old = &(pd->fv_olds[tid]);

  local_element_contributions lec_st;
  local_element_contributions *lec = &lec_st;

  basis_function *bfe[NUM_EQNS];

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

  bfe[VELOCITY1] = q2_bf;
  bfe[VELOCITY2] = q2_bf;
  bfe[PRESSURE] = q1_bf;
  bfe[AUX_VELOCITY1] = q2_bf;
  bfe[AUX_VELOCITY2] = q2_bf;
  bfe[AUX_PRESSURE] = q1_bf;

  for (int e = 0; e < NUM_EQNS; e++) {
    pd->elements[ielem].eqn_index[e] = e;
    pd->elements[ielem].eqn_dof[e] = bfe[e]->num_dof;
  }

  //    element_stiffness_pointers *esp = &elements[ielem].esp;

  zero_lec(lec);
  //    load_esp(esp, elements[ielem], bfe, mat_system, pd->mesh);

  int ip = 0;
  for (ip = 0; ip < MDE; ip++) {

      
    fv->wt = find_gauss_weight(bfe[VELOCITY1], ip);
    find_gauss_points(fv->xi, bfe[VELOCITY1], ip);
    for (int eqn = 0; eqn < NUM_EQNS; eqn++) {
      load_basis_functions(pd->mesh->dim, fv->xi, bfe[eqn]);
    }


    load_fv(pd->systems, fv, fv_old, bfe, &(pd->elements[ielem]));

    jacobian_inverse_and_mapping(pd->mesh, fv, bfe, pd->elements[ielem]);

    for (int eqn = 0; eqn < NUM_EQNS; eqn++) {
      load_basis_functions_grad(fv, bfe[eqn]);
    }

    load_fv_grad(pd->systems, fv, fv_old, bfe, &(pd->elements[ielem]));

    switch (matrix) {
    case 0:
      assemble_aux_u(pd->mesh->dim, fv, fv_old, bfe, lec,
		     pd->systems[matrix]->eqn_index,
		     0, 0.0000001, rho, mu);
      break;
    case 1:
      assemble_press_poisson(pd->mesh->dim, fv, fv_old, bfe, lec,
			     pd->systems[matrix]->eqn_index,
			     0, 0.000001, rho, mu);
      break;
    case 2:
      assemble_press_proj(pd->mesh->dim, fv, fv_old, bfe, lec,
			     pd->systems[matrix]->eqn_index,
			     0, 0.000001, rho, mu);
      break;
    case 3:
      assemble_press_update(pd->mesh->dim, fv, fv_old, bfe, lec,
			     pd->systems[matrix]->eqn_index,
			     0, 0.000001, rho, mu);
      break;
    default:
      break;
    }

  }

  set_dirichlet(pd->mesh, lec, pd->elements[ielem], pd->systems[matrix]->num_eqns, pd->systems[matrix]->eqn_index);
  load_lec_cpu(lec, pd->mesh, bfe, pd->elements[ielem], pd->systems[matrix]);
  //  load_lec_gpu(lec, pd->mesh, bfe, pd->elements[ielem], pd->systems[matrix]);
}

void matrix_fill_cpu(problem_data *pd, int matrix)
{
  for (int i = 0; i < pd->mesh->num_elem; i++) {
    matrix_fill_single(pd,
		       i,
		       matrix);
  }

}

void solve_umfpack(matrix_data *mat_system) {
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

void trilinos_solve(const Epetra_Comm & comm, matrix_data *mat_system);

void cusp_solve(matrix_data *mat_system)
{
  int m = mat_system->A->m;
  int nnz = mat_system->A->nnz;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  // *NOTE* raw pointers must be wrapped with thrust::device_ptr!
  thrust::device_ptr<int>   wrapped_device_Ap(mat_system->A->rowptr);
  thrust::device_ptr<int>   wrapped_device_Aj(mat_system->A->colind);
  thrust::device_ptr<double> wrapped_device_Ax(mat_system->A->val);
  thrust::device_ptr<double> wrapped_device_x(mat_system->delta_x);
  thrust::device_ptr<double> wrapped_device_y(mat_system->b);
  // use array1d_view to wrap the individual arrays
  typedef typename cusp::array1d_view< thrust::device_ptr<int>   > DeviceIndexArrayView;
  typedef typename cusp::array1d_view< thrust::device_ptr<double> > DeviceValueArrayView;
  DeviceIndexArrayView row_offsets   (wrapped_device_Ap, wrapped_device_Ap + m);
  DeviceIndexArrayView column_indices(wrapped_device_Aj, wrapped_device_Aj + nnz);
  DeviceValueArrayView values        (wrapped_device_Ax, wrapped_device_Ax + nnz);
  DeviceValueArrayView x (wrapped_device_x, wrapped_device_x + m);
  DeviceValueArrayView y (wrapped_device_y, wrapped_device_y + m);
  // combine the three array1d_views into a csr_matrix_view
  typedef cusp::csr_matrix_view<DeviceIndexArrayView,
    DeviceIndexArrayView,
    DeviceValueArrayView> DeviceView;
  DeviceView A(m, m, nnz, row_offsets, column_indices, values);
  typedef cusp::device_memory MemorySpace;
  cusp::monitor<double> monitor(y, 400, 1e-6, 0, false);
  // set preconditioner (identity)
  //  cusp::identity_operator<double, MemorySpace> M(A.num_rows, A.num_rows);
  // solve the linear system A * x = b with the BiConjugate Gradient Stabilized method
  start = std::chrono::system_clock::now();
  cusp::krylov::bicgstab(A, x, y, monitor);
  end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end-start;
 
  std::cout << " bicgstab: " << elapsed_seconds.count() << std::endl;

  monitor.print();

}

int solve_nonlinear(int num_newton_iterations, double residual_tolerance, double time, double dt,
		    problem_data *pd, const Epetra_Comm & comm, int matrix) {
  double theta = 0.5;
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
      zero_vec(pd->systems[matrix]->A->m, pd->systems[matrix]->delta_x);
      zero_vec(pd->systems[matrix]->A->m, pd->systems[matrix]->b);
      zero_vec(pd->systems[matrix]->A->nnz, pd->systems[matrix]->A->val);
      /* Matrix Fill */
      matrix_fill_gpu(time, dt, pd, matrix);
      //matrix_fill_cpu(pd, matrix);
      /* row sum scaling */
      row_sum_scale(pd->systems[matrix]);

      residual_L2_norm = l2_norm(pd->systems[matrix]->A->m, pd->systems[matrix]->b);
      /* Matrix solve */
      //qr_solve(mat_system);


      start = std::chrono::system_clock::now();

      //      solve_umfpack(mat_system);
    }

    if (comm.MyPID() == 0) {
      trilinos_solve(comm, pd->systems[matrix]);
    } else {
      trilinos_solve(comm, NULL);
    }
    //cusp_solve(pd->systems[matrix]);
    //test_bicgstab(pd, matrix);
    if (comm.MyPID() == 0) {
      end = std::chrono::system_clock::now();
 
      std::chrono::duration<double> elapsed_seconds = end-start;
 
      std::cout << " slv: " << elapsed_seconds.count() << std::endl;

      /* compute l2 norm */
      cudaDeviceSynchronize();

      update_L2_norm = l2_norm(pd->systems[matrix]->A->m, pd->systems[matrix]->delta_x);

      /* check converged */
      converged = residual_L2_norm < residual_tolerance;
      std::cout << newt_it << "\t" << residual_L2_norm << "\t"
		<< update_L2_norm << std::endl;

      for (int i = 0; i < pd->systems[matrix]->A->m; i++) {
	pd->systems[matrix]->x[i] -= pd->systems[matrix]->delta_x[i];
      }
    }

    comm.Broadcast(&converged, 1, 0);
    newt_it++;
  }
  return converged;
}
