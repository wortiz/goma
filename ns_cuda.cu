#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <stdlib.h>

extern "C" {
#include "exo_read_mesh.h"
}

#include <Epetra_config.h>
#include <Epetra_MpiComm.h>

#include "ns_constants.h"
#include "ns_structs.h"
#include "boundary_conditions.cuh"
#include "solve_problem.h"
int setup_problem(problem_data *pd);


int main(int argc, char **argv)
{
  std::string filename = "ldc.g";
  int num_bcs;
  problem_data *pd;
  boundary_condition *bcs;
  mesh_data *mesh;
  element *elements;
  MPI_Init (&argc, &argv);
  Epetra_MpiComm comm (MPI_COMM_WORLD);

  int rank = comm.MyPID();
  int error;

  if (rank == 0) {
    int nDevices;

    cudaMallocManaged(&pd, sizeof(problem_data));
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
      printf("  Memory Clock Rate (KHz): %d\n",
	     prop.memoryClockRate);
      printf("  Memory Bus Width (bits): %d\n",
	     prop.memoryBusWidth);
      printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
	     2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
    cudaSetDevice(0);

    mesh = exo_read_mesh(filename.c_str());

#if 1  /* LDC */
       num_bcs = 18;
       cudaMallocManaged(&elements, sizeof(element) * mesh->num_elem);
       cudaMallocManaged(&bcs, sizeof(boundary_condition) * num_bcs);

       /* Lid */
       set_boundary_condition(&bcs[0], 3, 1, DIRICHLET, VELOCITY1);
       set_boundary_condition(&bcs[1], 3, 0, DIRICHLET, VELOCITY2);
       set_boundary_condition(&bcs[2], 3, 1, DIRICHLET, AUX_VELOCITY1);
       set_boundary_condition(&bcs[3], 3, 0, DIRICHLET, AUX_VELOCITY2);

       set_boundary_condition(&bcs[4], 1, 0, DIRICHLET, VELOCITY1);
       set_boundary_condition(&bcs[5], 1, 0, DIRICHLET, VELOCITY2);
       set_boundary_condition(&bcs[6], 1, 0, DIRICHLET, AUX_VELOCITY1);
       set_boundary_condition(&bcs[7], 1, 0, DIRICHLET, AUX_VELOCITY2);

       set_boundary_condition(&bcs[8], 2, 0, DIRICHLET, VELOCITY1);
       set_boundary_condition(&bcs[9], 2, 0, DIRICHLET, VELOCITY2);
       set_boundary_condition(&bcs[10], 2, 0, DIRICHLET, AUX_VELOCITY1);
       set_boundary_condition(&bcs[11], 2, 0, DIRICHLET, AUX_VELOCITY2);

       set_boundary_condition(&bcs[12], 4, 0, DIRICHLET, VELOCITY1);
       set_boundary_condition(&bcs[13], 4, 0, DIRICHLET, VELOCITY2);
       set_boundary_condition(&bcs[14], 4, 0, DIRICHLET, AUX_VELOCITY1);
       set_boundary_condition(&bcs[15], 4, 0, DIRICHLET, AUX_VELOCITY2);

       set_boundary_condition(&bcs[16], 100, 0, DIRICHLET, PRESSURE);
       set_boundary_condition(&bcs[17], 100, 0, DIRICHLET, AUX_PRESSURE);

#else  /* STEP */
       num_bcs = 24;
       cudaMallocManaged(&elements, sizeof(element) * mesh->num_elem);
       cudaMallocManaged(&bcs, sizeof(boundary_condition) * num_bcs);

       /* Inflow */
       set_boundary_condition(&bcs[0], 4, 1, PARABOLIC, VELOCITY1);
       set_boundary_condition(&bcs[1], 4, 0, DIRICHLET, VELOCITY2);
       set_boundary_condition(&bcs[2], 4, 1, PARABOLIC, AUX_VELOCITY1);
       set_boundary_condition(&bcs[3], 4, 0, DIRICHLET, AUX_VELOCITY2);

       set_boundary_condition(&bcs[4], 5, 0, DIRICHLET, VELOCITY1);
       set_boundary_condition(&bcs[5], 5, 0, DIRICHLET, VELOCITY2);
       set_boundary_condition(&bcs[6], 5, 0, DIRICHLET, AUX_VELOCITY1);
       set_boundary_condition(&bcs[7], 5, 0, DIRICHLET, AUX_VELOCITY2);
       
       set_boundary_condition(&bcs[8], 6, 0, DIRICHLET, VELOCITY1);
       set_boundary_condition(&bcs[9], 6, 0, DIRICHLET, VELOCITY2);
       set_boundary_condition(&bcs[10], 6, 0, DIRICHLET, AUX_VELOCITY1);
       set_boundary_condition(&bcs[11], 6, 0, DIRICHLET, AUX_VELOCITY2);

       set_boundary_condition(&bcs[12], 3, 0, DIRICHLET, VELOCITY1);
       set_boundary_condition(&bcs[13], 3, 0, DIRICHLET, VELOCITY2);
       set_boundary_condition(&bcs[14], 3, 0, DIRICHLET, AUX_VELOCITY1);
       set_boundary_condition(&bcs[15], 3, 0, DIRICHLET, AUX_VELOCITY2);

       set_boundary_condition(&bcs[16], 1, 0, DIRICHLET, VELOCITY1);
       set_boundary_condition(&bcs[17], 1, 0, DIRICHLET, VELOCITY2);
       set_boundary_condition(&bcs[18], 1, 0, DIRICHLET, AUX_VELOCITY1);
       set_boundary_condition(&bcs[19], 1, 0, DIRICHLET, AUX_VELOCITY2);

       /* Outflow */
       set_boundary_condition(&bcs[20], 2, 0, DIRICHLET, VELOCITY2);
       set_boundary_condition(&bcs[21], 2, 0, DIRICHLET, AUX_VELOCITY2);
       set_boundary_condition(&bcs[22], 2, 0, DIRICHLET, PRESSURE);
       set_boundary_condition(&bcs[23], 2, 0, DIRICHLET, AUX_PRESSURE);
#endif
       
    cudaDeviceSynchronize();

    pd->bcs = bcs;
    pd->mesh = mesh;
    pd->elements = elements;
    pd->num_bcs = num_bcs;
    pd->num_elements = mesh->num_elem;
    pd->num_matrices = 4;
  
    error = setup_problem(pd);

    cudaDeviceSynchronize();
  }

  comm.Broadcast(&error, 1, 0);

  if (error == 0) {
    solve_problem(pd, comm);
  }

  MPI_Finalize();
  return error;
}

int setup_elements(problem_data *pd)
{
  mesh_data *mesh = pd->mesh;
  element *elements = pd->elements;
  boundary_condition *bcs = pd->bcs;
  int num_bcs = pd->num_bcs;
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
  size_t max = 0;
  for (int i = 0; i < mesh->num_elem; i++) {
    size_t count = elem_bcs[i].size();
    if (count > max) {
      max = count;
    }
  }
      

  if (max > MAX_BCS) {
    std::cerr << "Please increase MAX_BCS to " << max << std::endl;
    return -1;
  }

  for (int ielem = 0; ielem < mesh->num_elem; ielem++) {
    for (int i = 0; i < num_bcs; i++) {
      if (elem_bcs[ielem].find(bcs[i].dirichlet_index) != elem_bcs[ielem].end()) {
        elements[ielem].bcs[elements[ielem].num_bcs] = bcs[i];
        elements[ielem].num_bcs++;
      }
    }
  }

  return 0;
}

int setup_problem(problem_data *pd)
{
  int err = setup_elements(pd);
  cudaMallocManaged(&(pd->fvs),  sizeof(field_variables) * pd->mesh->num_elem);
  cudaMallocManaged(&(pd->fv_olds),  sizeof(field_variables) * pd->mesh->num_elem);
  return err;
}
