/*
 * cuda_interface.h
 *
 *  Created on: Aug 9, 2016
 *      Author: wortiz
 */

#ifndef CUDA_INTERFACE_H_
#define CUDA_INTERFACE_H_

typedef double dbl;

#include "rf_bc_const.h"

#ifdef __cplusplus
#include "cuda_macros.h"
#include "cuda_structs.cuh"
#include "Epetra_MpiComm.h"
extern "C" {
#include "exo_read_mesh.h"
}

namespace cuda {


struct cuda_data {
	boundary_condition *bcs;
	element *elements;
	mesh_data *mesh;
	Epetra_Comm &comm;
	int num_bcs;
	cuda_data(int num_bcs, mesh_data *mesh, element *elements,
			Epetra_MpiComm &comm, boundary_condition *bcs) :
			num_bcs(num_bcs), mesh(mesh), elements(elements), comm(comm), bcs(
					bcs) {
	}
	;
};



#else
typedef struct cuda_data cuda_data;
#endif

#ifdef __cplusplus
extern "C" {
#endif


int cuda_setup_problem(const char * exo_input_file, int num_bcs,
		struct Boundary_Condition * BC_Types, cuda_data **cuda_input_data);

int cuda_solve_problem(const char * exo_input_file,
		const char * exo_output_file, cuda_data *cuda_input_data);

#ifdef __cplusplus
} // extern "C"
} //namespace cuda
#endif

#endif /* CUDA_INTERFACE_H_ */
