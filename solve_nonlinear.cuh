/*
 * solve_nonlinear.cuh
 *
 *  Created on: Jul 5, 2016
 *      Author: wortiz
 */

#ifndef SOLVE_NONLINEAR_CUH_
#define SOLVE_NONLINEAR_CUH_
#include <iostream>

#include "ns_structs.h"
#include "device_functions.h"
__host__ __device__ int index_solution(matrix_data *mat_system, int node, int eqn);

int solve_nonlinear(int num_newton_iterations, double residual_tolerance, double time, double dt,
                    problem_data *pd, const Epetra_Comm & comm, int matrix);

#endif /* SOLVE_NONLINEAR_CUH_ */
