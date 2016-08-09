/*
 * cuda_solve_nonlinear.cuh
 *
 *  Created on: Aug 9, 2016
 *      Author: wortiz
 */

#ifndef CUDA_SOLVE_NONLINEAR_CUH_
#define CUDA_SOLVE_NONLINEAR_CUH_

#include "cuda_interface.h"

namespace cuda {

__host__ __device__ int index_solution(problem_data *mat_system, int node,
		int eqn);

int cuda_solve_nonlinear(int num_newton_iterations, double residual_tolerance,
		basis_function** bfs, element *elements, problem_data *mat_system,
		mesh_data *mesh, const Epetra_Comm & comm);

}
#endif /* CUDA_SOLVE_NONLINEAR_CUH_ */
