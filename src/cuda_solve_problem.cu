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

#include "cuda_solve_nonlinear.cuh"

#include "cuda_structs.cuh"

#include "cuda_interface.h"
#include <cuda_runtime.h>

namespace cuda {

static int in_list(int start, int end, int value, int *list) {
	for (int i = start; i < end; i++) {
		if (list[i] == value) {
			return i;
		}
	}
	return -1;
}

int eqn_num_dof[3] = { 9, 9, 4 };

void matrix_setup(mesh_data *mesh, element *elements, problem_data *mat_system,
		basis_function **bfs) {
	int num_global_elements = 0;

	bool *nodes = new bool[mesh->num_nodes];
	for (int i = 0; i < mesh->num_nodes; i++) {
		nodes[i] = true;
	}
	for (int el = 0; el < mesh->num_elem; el++) {
		for (int n = 0; n < CUDA_MDE; n++) {
			int gnn = elements[el].gnn[n] - 1;
			if (nodes[gnn]) {
				for (int e = 0; e < CUDA_NUM_EQNS; e++) {
					if (n < eqn_num_dof[e]) {
						num_global_elements++;
					}
				}
				nodes[gnn] = false;
			}
		}
	}

	delete nodes;

	cudaMallocManaged(&mat_system->x, sizeof(double) * num_global_elements);

	for (int i = 0; i < num_global_elements; i++) {
		mat_system->x[i] = 0;
	}

	cudaMallocManaged(&mat_system->b, sizeof(double) * num_global_elements);
	cudaMallocManaged(&mat_system->delta_x,
			sizeof(double) * num_global_elements);

	mat_system->A = new CrsMatrix;
	cudaMallocManaged(&mat_system->A, sizeof(CrsMatrix));

	cudaMallocManaged(&mat_system->A->rowptr,
			sizeof(int) * (num_global_elements + 1));
	mat_system->A->rowptr[0] = 0;

	std::vector<int> colind;
	mat_system->A->nnz = 0;

	int global_row = 0;
	for (int node = 0; node < mesh->num_nodes; node++) {
		for (int e = 0; e < CUDA_NUM_EQNS; e++) {

			int global_row_index = index_solution(mat_system, node, e);

			if (global_row_index != -1) {
				assert(global_row == global_row_index);

				mat_system->A->rowptr[global_row + 1] =
						mat_system->A->rowptr[global_row];

				int elidx = 0;
				while (elidx < mesh->max_elem_per_node
						&& mesh->node_elem[node][elidx] != 0) {
					int elem = mesh->node_elem[node][elidx] - 1;
					for (int inter_node = 0; inter_node < CUDA_MDE;
							inter_node++) {
						int gnn = elements[elem].gnn[inter_node] - 1;
						for (int v = 0; v < CUDA_NUM_EQNS; v++) {
							if (inter_node < eqn_num_dof[v]) {
								int global_col = index_solution(mat_system, gnn,
										v);
								if (in_list(mat_system->A->rowptr[global_row],
										mat_system->A->rowptr[global_row + 1],
										global_col, &colind[0]) == -1) {
									colind.push_back(global_col);
									mat_system->A->rowptr[global_row + 1] += 1;
									mat_system->A->nnz += 1;
								}
							}
						}
					}
					elidx++;
				}

				std::sort(colind.begin() + mat_system->A->rowptr[global_row],
						colind.begin() + mat_system->A->rowptr[global_row + 1]);

				global_row++;
			}
		}
	}

	mat_system->A->m = num_global_elements;
	cudaMallocManaged(&mat_system->A->val, sizeof(double) * mat_system->A->nnz);
	cudaMallocManaged(&mat_system->A->colind,
			sizeof(double) * mat_system->A->nnz);
	assert(colind.size() == mat_system->A->nnz);
	for (int i = 0; i < mat_system->A->nnz; i++) {
		mat_system->A->colind[i] = colind[i];
	}
}

void set_dirichlet(mesh_data *mesh, problem_data *mat_system,
		boundary_condition *bcs, int num_bcs) {
	for (int i = 0; i < num_bcs; i++) {
		int didx;
		switch (bcs[i].type) {
		case CUDA_DIRICHLET:
			for (didx = 0; didx < mesh->num_dirichlet; didx++) {
				int id = bcs[i].dirichlet_index;
				if (id == mesh->dirichlet_ids[didx]) {
					break;
				}
			}

			for (int node = mesh->dirichlet_node_index[didx];
					node < mesh->dirichlet_node_index[didx + 1]; node++) {
				int gnn = mesh->dirichlet_node_list[node];
				int idx = index_solution(mat_system, gnn - 1, bcs[i].eqn);
				mat_system->x[idx] = bcs[i].value;
			}
			break;
		case CUDA_PARABOLIC:
			for (didx = 0; didx < mesh->num_dirichlet; didx++) {
				int id = bcs[i].dirichlet_index;
				if (id == mesh->dirichlet_ids[didx]) {
					break;
				}
			}

			for (int node = mesh->dirichlet_node_index[didx];
					node < mesh->dirichlet_node_index[didx + 1]; node++) {
				int gnn = mesh->dirichlet_node_list[node] - 1;
				int idx = index_solution(mat_system, gnn, bcs[i].eqn);
				double yloc = mesh->coord[1][gnn];
				mat_system->x[idx] = bcs[i].value * (1 - yloc * yloc);
			}
			break;
		default:
			break;
		}
	}
}

void write_results(const char * exo_input_file, const char * exo_output_file,
		mesh_data *mesh, problem_data *mat_system, element *elements) {
	std::ofstream myfile;
	myfile.open("results.txt");
	double ** values = new double*[CUDA_NUM_EQNS];

	for (int e = 0; e < CUDA_NUM_EQNS; e++) {
		values[e] = new double[mesh->num_nodes];
	}

	for (int i = 0; i < mesh->num_elem; i++) {
		for (int e = 0; e < CUDA_NUM_EQNS; e++) {
			for (int j = 0; j < CUDA_MDE; j++) {
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

	exo_write_results(mesh, exo_input_file, exo_output_file, values);
}

extern "C" int cuda_solve_problem(const char * exo_input_file,
		const char * exo_output_file, cuda_data *cuda_input_data) {

	int num_bcs = cuda_input_data->num_bcs;
	mesh_data *mesh = cuda_input_data->mesh;
	boundary_condition *bcs = cuda_input_data->bcs;
	element *elements = cuda_input_data->elements;

	double residual_tolerance = 1e-10;
	int num_newton_iterations = 10;

	problem_data *mat_system = new problem_data;
	Epetra_MpiComm comm(MPI_COMM_WORLD);
	if (comm.MyPID() == 0) {
		cudaMallocManaged(&mat_system, sizeof(problem_data));
		cudaMallocManaged(&mat_system->solution_index,
				sizeof(int *) * CUDA_NUM_EQNS);
		for (int i = 0; i < CUDA_NUM_EQNS; i++) {
			cudaMallocManaged(&mat_system->solution_index[i],
					sizeof(int) * mesh->num_nodes);
		}

		for (int e = 0; e < CUDA_NUM_EQNS; e++) {
			for (int i = 0; i < mesh->num_nodes; i++) {
				mat_system->solution_index[e][i] = -1;
			}
		}

		int index = 0;

		for (int i = 0; i < mesh->num_elem; i++) {
			for (int j = 0; j < CUDA_MDE; j++) {
				int gnn = elements[i].gnn[j] - 1;
				for (int e = 0; e < CUDA_NUM_EQNS; e++) {
					if (j < eqn_num_dof[e]
							&& mat_system->solution_index[e][gnn] == -1) {
						mat_system->solution_index[e][gnn] = index;
						index++;
					}
				}
			}
		}

		/* Set up matrix */
		matrix_setup(mesh, elements, mat_system, NULL);

		cudaDeviceSynchronize();

		/* Initialize Dirichlet conditions */
		set_dirichlet(mesh, mat_system, bcs, num_bcs);

		cudaDeviceSynchronize();
		/* Newton's method */
	}

	cuda_solve_nonlinear(num_newton_iterations, residual_tolerance, NULL,
			elements, mat_system, mesh, comm);

	if (comm.MyPID() == 0) {
		write_results(exo_input_file, exo_output_file, mesh, mat_system,
				elements);
	}

	return 0;
}

}
