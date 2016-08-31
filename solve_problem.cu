/*
 * solve_problem.cpp
 *
 *  Created on: Jul 5, 2016
 *      Author: wortiz
 */

#include <iostream>
#include <fstream>

#include <vector>
#include <algorithm>
#include <assert.h>
extern "C" {
#include "exo_read_mesh.h"
}

#include "boundary_conditions.cuh"
#include "solve_nonlinear.cuh"

#include "ns_structs.h"

#include <cuda_runtime.h>

static int in_list(int start, int end, int value, int *list)
{
  for (int i = start; i < end; i++) {
    if (list[i] == value) {
      return i;
    }
  }
  return -1;
}

int eqn_num_dof[NUM_EQNS] = {9,9,4,9,9,4};

void matrix_setup(mesh_data *mesh, element *elements, matrix_data *mat_system) {
		  
  int num_global_elements = 0;

  bool *nodes = new bool[mesh->num_nodes];
  for (int i = 0; i < mesh->num_nodes; i++) {
    nodes[i] = true;
  }
  for (int el = 0; el < mesh->num_elem; el++) {
    for (int n = 0; n < MDE; n++) {
      int gnn = elements[el].gnn[n] - 1;
      if (nodes[gnn]) {
	for (int e = 0; e < mat_system->num_eqns; e++) {
	  int eqn = mat_system->inv_eqn_index[e];
	  if (n < eqn_num_dof[eqn]) {
	    num_global_elements++;
	  }
	}
	nodes[gnn] = false;
      }
    }
  }

  delete[] nodes;

  cudaMallocManaged(&mat_system->x, sizeof(double) * num_global_elements);
  cudaMallocManaged(&mat_system->x_old, sizeof(double) * num_global_elements);
  memset(mat_system->x_old, 0, sizeof(double) * num_global_elements);

  for (int i = 0; i < num_global_elements; i++) {
    mat_system->x[i] = 0;
  }
  
  cudaMallocManaged(&mat_system->b, sizeof(double) * num_global_elements);
  cudaMallocManaged(&mat_system->delta_x, sizeof(double) * num_global_elements);

  mat_system->A = new CrsMatrix;
  cudaMallocManaged(&mat_system->A, sizeof(CrsMatrix));
  
  cudaMallocManaged(&mat_system->A->rowptr, sizeof(int) * (num_global_elements+1));
  mat_system->A->rowptr[0] = 0;

  std::vector<int> colind;
  mat_system->A->nnz = 0;

  int global_row = 0;
  for (int node = 0; node < mesh->num_nodes; node++) {
    for (int e = 0; e < mat_system->num_eqns; e++) {
      int eqn = mat_system->inv_eqn_index[e];
      assert(eqn != -1);
      int global_row_index = index_solution(mat_system, node, eqn);

      if (global_row_index != -1) {
	assert(global_row == global_row_index);

	mat_system->A->rowptr[global_row+1] = mat_system->A->rowptr[global_row];
      
	int elidx = 0;
	while (elidx < mesh->max_elem_per_node
	       && mesh->node_elem[node][elidx] != 0) {
	  int elem = mesh->node_elem[node][elidx] - 1;
	  for (int inter_node = 0; inter_node < MDE; inter_node++) {
	    int gnn = elements[elem].gnn[inter_node] - 1;
	    for (int v = 0; v < mat_system->num_eqns; v++) {
	      int var = mat_system->inv_eqn_index[v];
	      assert(var != -1);
	      if (inter_node < eqn_num_dof[var]) {
		int global_col = index_solution(mat_system, gnn, var);
		if (in_list(mat_system->A->rowptr[global_row],
			    mat_system->A->rowptr[global_row+1],
			    global_col, &colind[0]) == -1) {
		  colind.push_back(global_col);
		  mat_system->A->rowptr[global_row+1] += 1;
		  mat_system->A->nnz += 1;
		  std::sort(colind.begin() + mat_system->A->rowptr[global_row],
			    colind.begin() + mat_system->A->rowptr[global_row+1]);

		}
	      }
	    }
	  }
	  elidx++;
	}
	
	global_row++;
      }
    }
  }

  mat_system->A->m = num_global_elements;
  cudaMallocManaged(&mat_system->A->val, sizeof(double) * mat_system->A->nnz);
  cudaMallocManaged(&mat_system->A->colind, sizeof(double) * mat_system->A->nnz);
  assert(colind.size() == mat_system->A->nnz);
  for (int i = 0; i < mat_system->A->nnz; i++) {
    mat_system->A->colind[i] = colind[i];
  }
}

void set_dirichlet(problem_data *pd, int matrix)
{
  matrix_data *mat_system = pd->systems[matrix];
  for (int i = 0; i < pd->num_bcs; i++) {
    if (mat_system->eqn_index[pd->bcs[i].eqn] != -1) {
      int didx;
      switch (pd->bcs[i].type) {
      case DIRICHLET:
	for (didx = 0; didx < pd->mesh->num_dirichlet; didx++) {
	  int id = pd->bcs[i].dirichlet_index;
	  if (id == pd->mesh->dirichlet_ids[didx]) {
	    break;
	  }
	}

	for (int node = pd->mesh->dirichlet_node_index[didx];
	     node < pd->mesh->dirichlet_node_index[didx + 1]; node++) {
	  int gnn = pd->mesh->dirichlet_node_list[node];
	  int idx = index_solution(mat_system, gnn - 1, pd->bcs[i].eqn);
	  mat_system->x[idx] = pd->bcs[i].value;
	}
	break;
      case PARABOLIC:
	for (didx = 0; didx < pd->mesh->num_dirichlet; didx++) {
	  int id = pd->bcs[i].dirichlet_index;
	  if (id == pd->mesh->dirichlet_ids[didx]) {
	    break;
	  }
	}

	for (int node = pd->mesh->dirichlet_node_index[didx];
	     node < pd->mesh->dirichlet_node_index[didx + 1]; node++) {
	  int gnn = pd->mesh->dirichlet_node_list[node] -1;
	  int idx = index_solution(mat_system, gnn, pd->bcs[i].eqn);
	  double yloc = pd->mesh->coord[1][gnn];
	  mat_system->x[idx] = pd->bcs[i].value * (1 - yloc*yloc);
	}
      default:
	break;
      }
    }
  }
}

void write_results(double time, problem_data *pd)
{
  mesh_data *mesh = pd->mesh;
  element *elements = pd->elements;
  double ** values = new double*[3];

  for (int e = 0; e < 3; e++) {
    values[e] = new double[mesh->num_nodes];
  }

  for (int i = 0; i < mesh->num_elem; i++) {
    for (int e = 0; e < NUM_EQNS; e++) {
      matrix_data *mat_system;
      switch (e) {
      case VELOCITY1:
      case VELOCITY2:
	mat_system = pd->systems[2];
	break;
      case PRESSURE:
	mat_system = pd->systems[3];
	break;
      default:
	continue;
      }
      for (int j = 0; j < MDE; j++) {
	int gnn = elements[i].gnn[j] - 1;
	if (e == PRESSURE && j > 3) {
	  int ileft = j - 4;
	  int iright = j - 3;
	  int gnnl = elements[i].gnn[ileft] - 1;
	  int gnnr = elements[i].gnn[iright] - 1;
	  if (j < 8) {
	    values[e][gnn] = 0.5
	      * (values[e][gnnl] + values[e][gnnr]);
	  } else {
	    values[e][gnn] = 0.0;
	    for (ileft = 0; ileft < 4; ileft++) {
	      gnnl = elements[i].gnn[ileft] - 1;
	      values[e][gnn] += 0.25 * values[e][gnnl];
	    }
	  }
	} else {
	  int idx = index_solution(mat_system, gnn, e);
	  values[e][gnn] = mat_system->x[idx];
	}
      }
    }

  }

  std::string outfile = "out.e";
  std::string infile = "ldc.g";

  exo_write_results(time, mesh, &infile[0], &outfile[0], values);
}

void setup_cbs_matrices(mesh_data *mesh, element *elements, matrix_data **mat_systems)
{

  for (int matrix = 0; matrix < 4; matrix++) {
    matrix_data *mat_system;
    cudaMallocManaged(&mat_system, sizeof(matrix_data));
    
    for (int i = 0; i < NUM_EQNS; i++) {
      mat_system->eqn_index[i] = -1;
      mat_system->inv_eqn_index[i] = -1;
    }

    switch (matrix) {
    case 0:
      mat_system->num_eqns = 2;
      mat_system->eqn_index[AUX_VELOCITY1] = 0;
      mat_system->eqn_index[AUX_VELOCITY2] = 1;
      mat_system->inv_eqn_index[0] = AUX_VELOCITY1;
      mat_system->inv_eqn_index[1] = AUX_VELOCITY2;
      break;
    case 1:
      mat_system->num_eqns = 1;
      mat_system->eqn_index[AUX_PRESSURE] = 0;
      mat_system->inv_eqn_index[0] = AUX_PRESSURE;
      break;
    case 2:
      mat_system->num_eqns = 2;
      mat_system->eqn_index[VELOCITY1] = 0;
      mat_system->eqn_index[VELOCITY2] = 1;
      mat_system->inv_eqn_index[0] = VELOCITY1;
      mat_system->inv_eqn_index[1] = VELOCITY2;
      break;
    case 3:
      mat_system->num_eqns = 1;
      mat_system->eqn_index[PRESSURE] = 0;
      mat_system->inv_eqn_index[0] = PRESSURE;
      break;
    default:
      std::cerr << "Unknown error occurred" << std::endl;
      break;
    }
    
    cudaMallocManaged(&mat_system->solution_index, sizeof(int *) * mat_system->num_eqns);
    for (int i = 0; i < mat_system->num_eqns; i++) {
      cudaMallocManaged(&mat_system->solution_index[i],  sizeof(int)*mesh->num_nodes);
    }
  

    for (int e = 0; e < mat_system->num_eqns; e++) {
      for (int i = 0; i < mesh->num_nodes; i++) {
	mat_system->solution_index[e][i] = -1;
      }
    }
  
    int index = 0;

    for (int i = 0; i < mesh->num_elem; i++) {
      for (int j = 0; j < MDE; j++) {
	int gnn = elements[i].gnn[j] - 1;
	for (int e = 0; e < mat_system->num_eqns; e++) {
	  int eqn = mat_system->inv_eqn_index[e];
	  if (j < eqn_num_dof[eqn]
	      && mat_system->solution_index[e][gnn] == -1) {
	    mat_system->solution_index[e][gnn] = index;
	    index++;
	  }
	}
      }
    }

    /* Set up matrix */
    matrix_setup(mesh, elements, mat_system);
    mat_systems[matrix] = mat_system;
  }

}


int solve_problem(problem_data *pd, const Epetra_Comm & comm) {
  double residual_tolerance = 1e-9;
  int num_newton_iterations = 10;
  double time = 0;
  int time_steps = 30;
  double dt = 1e-4;

  if (comm.MyPID() == 0) {

    cudaMallocManaged(&(pd->systems), sizeof(matrix_data *) * 4);

    setup_cbs_matrices(pd->mesh, pd->elements, pd->systems);

    cudaDeviceSynchronize();

    /* Newton's method */
  }

  if (comm.MyPID() == 0) {
    /* Initialize Dirichlet conditions */
    set_dirichlet(pd, 0);
    cudaDeviceSynchronize();
  }  

  if (comm.MyPID() == 0) {
    /* Initialize Dirichlet conditions */
    set_dirichlet(pd, 1);

    cudaDeviceSynchronize();
  }

  if (comm.MyPID() == 0) {
    /* Initialize Dirichlet conditions */
    set_dirichlet(pd, 2);

    cudaDeviceSynchronize();
  }

  if (comm.MyPID() == 0) {
    /* Initialize Dirichlet conditions */
    set_dirichlet(pd, 3);

    cudaDeviceSynchronize();
  }

  for (int i = 0; i < time_steps; i++) {
    int converged;

    time = time + dt;
    
    if (comm.MyPID() == 0) {
    std::cout << "Time step " << i << " at time " << time << " with dt = " << dt << std::endl;


      for (int i = 0; i < pd->num_matrices; i++) {
	set_dirichlet(pd, i);
	for (int j = 0; j < pd->systems[i]->A->m; j++) {
	  pd->systems[i]->x_old[j] = pd->systems[i]->x[j];
	}
      }
    }  

    for (int matrix = 0; matrix < 4; matrix++) {
      if (comm.MyPID() == 0) {
	std::cout << " Matrix = " << matrix << std::endl;
      
    
	/* Initialize Dirichlet conditions */
	set_dirichlet(pd, matrix);

	cudaDeviceSynchronize();
      }
      converged = solve_nonlinear(num_newton_iterations, residual_tolerance, time, dt, pd, comm, matrix);

      if (!converged) {
	return -1;
      }

    }
    
    if (comm.MyPID() == 0) {
      std::cout << "\nTime step complete." << std::endl;

      write_results(time, pd);
    }

  }
  return 0;
}
