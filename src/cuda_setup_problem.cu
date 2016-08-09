/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <vector>
#include <set>

#include "cuda_interface.h"

namespace cuda {

static void setup_elements(mesh_data *mesh, boundary_condition *bcs, int num_bcs,
                    element *elements);

int translate_boundary_conditions(int num_bcs,
		struct Boundary_Condition *BC_Types, boundary_condition *bcs) {

	for (int i = 0; i < num_bcs; i++) {

		if (BC_Types[i].desc->method != DIRICHLET && BC_Types[i].desc->method != CUDA_PARABOLA) {
			std::cerr << "Can only use dirichlet boundary conditions with CUDA atm." << std::endl;
			return -1;
		}

		int equation = BC_Types[i].desc->equation;

		switch (equation) {
		case VELOCITY1:
			bcs[i].eqn = CUDA_VELOCITY1;
			break;
		case VELOCITY2:
			bcs[i].eqn = CUDA_VELOCITY2;
			break;
		case PRESSURE:
			bcs[i].eqn = CUDA_PRESSURE;
			break;
		default:
			std::cerr << "Only u, v, p supported" << std::endl;
			return -1;
		}
		bcs[i].dirichlet_index = BC_Types[i].BC_ID;
		bcs[i].type = CUDA_DIRICHLET;
		bcs[i].value = BC_Types[i].BC_Data_Float[0];

		if (BC_Types[i].desc->method == CUDA_PARABOLA) {
			bcs[i].type = CUDA_PARABOLIC;
		}
	}

	return 0;

}

extern "C" int cuda_setup_problem(const char * exo_input_file, int num_bcs,
		struct Boundary_Condition * BC_Types, cuda_data **cuda_input_data) {
	boundary_condition *bcs;
	mesh_data *mesh;
	element *elements;
	Epetra_MpiComm comm(MPI_COMM_WORLD);

	int rank = comm.MyPID();
	int status = 0;

	if (rank == 0) {
		int nDevices;

		cudaGetDeviceCount(&nDevices);
		for (int i = 0; i < nDevices; i++) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			printf("Device Number: %d\n", i);
			printf("  Device name: %s\n", prop.name);
			printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
			printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
			printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
					2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8)
							/ 1.0e6);
		}
		cudaSetDevice(0);

		mesh = exo_read_mesh(exo_input_file);

		cudaMallocManaged(&elements, sizeof(element) * mesh->num_elem);
		cudaMallocManaged(&bcs, sizeof(boundary_condition) * num_bcs);

		status = translate_boundary_conditions(num_bcs, BC_Types, bcs);

		if (status == 0) {
			cudaDeviceSynchronize();

			setup_elements(mesh, bcs, num_bcs, elements);

			cudaDeviceSynchronize();
		}
	}

	comm.Broadcast(&status, 1, 0);

	if (status != 0) return status;

	*cuda_input_data = new cuda_data(num_bcs, mesh, elements, comm, bcs);

	if (comm.MyPID() == 0) {
		std::cout << "Num_BCS = " << (*cuda_input_data)->num_bcs << std::endl;
		std::cout << "Elems = " << mesh->num_elem << std::endl;
		std::cout << "Nodes = " << mesh->num_nodes << std::endl << std::endl;
	}

	return 0;
}

static void setup_elements(mesh_data *mesh, boundary_condition *bcs, int num_bcs,
                    element *elements)
{

  for (int mat = 0; mat < mesh->num_mat; mat++) {
    for (int ielem = 0; ielem < mesh->num_elem; ielem++) {
      elements[ielem].id = ielem;
      for (int lnn = 0; lnn < mesh->num_nodes_per_elem[mat]; lnn++) {
        elements[ielem].gnn[lnn] = mesh->connect[mat][ielem
            * mesh->num_nodes_per_elem[mat] + lnn];
      }
    }
  }

  std::vector< std::vector< boundary_condition > > dirichlet_to_bcs;

  for (int ielem = 0; ielem < mesh->num_elem; ielem++) {
    elements[ielem].num_bcs = 0;
  }

  for (int didx = 0; didx < mesh->num_dirichlet; didx++) {
    std::vector< boundary_condition > matching_bcs;
    for (int i = 0; i < num_bcs; i++) {
      int id = bcs[i].dirichlet_index;
      if (id == mesh->dirichlet_ids[didx]) {
        matching_bcs.push_back(bcs[i]);
      }
    }
    dirichlet_to_bcs.push_back(matching_bcs);
  }

  std::vector< std::set< int > > elem_bcs(mesh->num_elem);

  for (int didx = 0; didx < mesh->num_dirichlet; didx++) {
    int did = mesh->dirichlet_ids[didx];
    for (int node = mesh->dirichlet_node_index[didx];
        node < mesh->dirichlet_node_index[didx + 1]; node++) {

      int gnn = mesh->dirichlet_node_list[node];
      for (int i = 0; i < mesh->max_elem_per_node; i++) {
        if (mesh->node_elem[gnn - 1][i] > 0) {
          int elem = mesh->node_elem[gnn - 1][i] - 1;
          elem_bcs[elem].insert(did);
        }
      }
    }
  }

  for (int ielem = 0; ielem < mesh->num_elem; ielem++) {
    for (int i = 0; i < num_bcs; i++) {
      if (elem_bcs[ielem].find(bcs[i].dirichlet_index) != elem_bcs[ielem].end()) {
        elements[ielem].bcs[elements[ielem].num_bcs] = bcs[i];
        elements[ielem].num_bcs++;
      }
    }
  }
}

} //namespace cuda
