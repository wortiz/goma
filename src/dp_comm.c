/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) 2022 Goma Developers, National Technology & Engineering   *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

#include "dp_comm.h"
#include "dp_map_comm_vec.h"
#include "dp_types.h"
#include "dpi.h"
#include "el_elm_info.h"
#include "rf_allo.h"
#include "rf_fem.h"

/* System Include files */

/* User include files */
/*
#include "el_elm.h"
#include "std.h"

#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "rf_bc_const.h"
#include "rf_masks.h"

#include "mm_eh.h"

#include "dp_types.h"
#include "dpi.h"
#include "exo_struct.h"
#include <cstdlib>
#include <mpi.h>
#include <petsclog.h>
*/

#define GOMA_DP_COMM_C

/********************************************************************/
/********************************************************************/
/********************************************************************/
void exchange_elem(Exo_DB *exo, Dpi *dpi, dbl *x) {

  if (Num_Proc == 1)
    return;

  // Just going to use a dumb exchange right now assuming this isn't performance
  // critical and is only called a few times in the code.
  //
  // Given our current element decomposition we should only need to send to
  // neighbors with a higher procid Just send the whole list for now, elem count
  // should be much less than dof so this shouldn't be too bad

  // Find elements and their ids in base mesh
  int n_elems = exo->base_mesh->num_elems;
  int *base_mesh_elem_ids = calloc(n_elems, sizeof(int));
  dbl *base_mesh_elem_vals = calloc(n_elems, sizeof(dbl));

  int proc_elem = 0;
  int base_elem = 0;
  for (int i = 0; i < exo->num_elem_blocks; i++) {
    for (int j = 0; j < exo->eb_num_elems[i]; j++) {
      int index = exo->eb_ghost_elem_to_base[i][j];
      if (index != -1) {
        base_mesh_elem_ids[base_elem] = dpi->elem_index_global[proc_elem];
        base_mesh_elem_vals[base_elem] = x[proc_elem];
        base_elem++;
      }
      proc_elem++;
    }
  }

  GOMA_ASSERT(proc_elem == exo->num_elems);
  GOMA_ASSERT(base_elem == exo->base_mesh->num_elems);


  int *recv_from_neighbor_sz = calloc(dpi->num_neighbors, sizeof(int));
  MPI_Request *requests = calloc(dpi->num_neighbors * 4, sizeof(MPI_Request));

  // get element counts of neighbors
  int n_req = 0;
  for (int i = 0; i < dpi->num_neighbors; i++) {
      MPI_Irecv(&recv_from_neighbor_sz[i], 1, MPI_INT, dpi->neighbor[i], 801, MPI_COMM_WORLD,
                &requests[n_req++]);
      MPI_Isend(&n_elems, 1, MPI_INT, dpi->neighbor[i], 801, MPI_COMM_WORLD,
                &requests[n_req++]);
  }

  MPI_Waitall(n_req, requests, MPI_STATUSES_IGNORE);

  int **recv_ids = calloc(dpi->num_neighbors, sizeof(int *));
  double **recv_vals = calloc(dpi->num_neighbors, sizeof(double *));

  n_req = 0;
  for (int i = 0; i < dpi->num_neighbors; i++) {
      recv_ids[i] = calloc(recv_from_neighbor_sz[i], sizeof(int));
      recv_vals[i] = calloc(recv_from_neighbor_sz[i], sizeof(double));

      MPI_Irecv(recv_ids[i], recv_from_neighbor_sz[i], MPI_INT, dpi->neighbor[i], 802,
                MPI_COMM_WORLD, &requests[n_req++]);
      MPI_Irecv(recv_vals[i], recv_from_neighbor_sz[i], MPI_DOUBLE, dpi->neighbor[i], 803,
                MPI_COMM_WORLD, &requests[n_req++]);
      MPI_Isend(base_mesh_elem_ids, n_elems, MPI_INT, dpi->neighbor[i], 802,
                MPI_COMM_WORLD, &requests[n_req++]);
      MPI_Isend(base_mesh_elem_vals, n_elems, MPI_DOUBLE, dpi->neighbor[i], 803, MPI_COMM_WORLD,
                &requests[n_req++]);
  }

  MPI_Waitall(n_req, requests, MPI_STATUSES_IGNORE);

  // We probably only have to do this a few times overall so complexity probably won't hurt,
  // if it does switch to sorted ids or a better algorithm

  for (int i = 0; i < dpi->num_neighbors; i++) {
    for (int j = 0; j < recv_from_neighbor_sz[i]; j++) {
      int id = recv_ids[i][j];
      int index = in_list(id, 0, exo->num_elems, dpi->elem_index_global);
      if (index != -1) {
        x[index] = recv_vals[i][j];
      }
    }
  }

  for (int i = 0; i < dpi->num_neighbors; i++) {
    free(recv_ids[i]);
    free(recv_vals[i]);
  }

  free(base_mesh_elem_ids);
  free(base_mesh_elem_vals);

  free(recv_ids);
  free(recv_vals);
  free(recv_from_neighbor_sz);
}

void exchange_dof(Comm_Ex *cx, Dpi *dpi, double *x, int imtrx)

/************************************************************
 *
 *  exchange_dof():
 *
 *  send/recv appropriate pieces of a dof-based double array
 ************************************************************/
{
  COMM_NP_STRUCT *np_base, *np_ptr;
  double *ptr_send_list, *ptr_recv_list;
  register double *ptrd;
  register int *ptr_int, i;
  int p;
  int num_neighbors = dpi->num_neighbors;
  int total_num_send_unknowns;

  if (num_neighbors == 0)
    return;

#ifdef PARALLEL
  total_num_send_unknowns = ptr_dof_send[imtrx][dpi->num_neighbors];
  np_base = alloc_struct_1(COMM_NP_STRUCT, dpi->num_neighbors);
  ptrd = (double *)alloc_dbl_1(total_num_send_unknowns, DBL_NOINIT);
  ptr_send_list = ptrd;

  /*
   * gather up the list of send unknowns
   */
  ptr_int = list_dof_send[imtrx];
  for (i = total_num_send_unknowns; i > 0; i--) {
    *ptrd++ = x[*ptr_int++];
  }

  /*
   * store base address for the start of the external degrees of freedom
   * in this vector
   */
  ptr_recv_list = x + num_internal_dofs[imtrx] + num_boundary_dofs[imtrx];

  np_ptr = np_base;
  for (p = 0; p < dpi->num_neighbors; p++) {
    np_ptr->neighbor_ProcID = cx[p].neighbor_name;
    np_ptr->send_message_buf = (void *)(ptr_send_list + ptr_dof_send[imtrx][p]);
    np_ptr->send_message_length = sizeof(double) * cx[p].num_dofs_send;
    np_ptr->recv_message_buf = (void *)ptr_recv_list;
    np_ptr->recv_message_length = sizeof(double) * cx[p].num_dofs_recv;
    ptr_recv_list += cx[p].num_dofs_recv;
    np_ptr++;
  }
  exchange_neighbor_proc_info(dpi->num_neighbors, np_base);
  safer_free((void **)&np_base);
  safer_free((void **)&ptr_send_list);
#endif /* PARALLEL */
}
/********************************************************************/
/********************************************************************/
/********************************************************************/
/*
{
#ifdef PARALLEL
  int p;

  for ( p=0; p<d->num_neighbors; p++)
    {
      MPI_Irecv(a,
                1,
                cx[p].mpidt_d_dof_recv,
                cx[p].neighbor_name,
                555,
                MPI_COMM_WORLD,
                Request + Num_Requests*p + 2 );

      MPI_Isend(a,
                1,
                cx[p].mpidt_d_dof_send,
                cx[p].neighbor_name,
                555,
                MPI_COMM_WORLD,
                ( Request + Num_Requests*p + 3 ) );
    }

  for ( p=0; p<d->num_neighbors; p++)
    {
      MPI_Wait( Request + Num_Requests*p + 2, Status  + Num_Requests*p + 2 );
      MPI_Wait( Request + Num_Requests*p + 3, Status  + Num_Requests*p + 3 );
    }
#endif

  return;
}
*/
/********************************************************************/
/********************************************************************/
/********************************************************************/

void exchange_node(Comm_Ex *cx, Dpi *dpi, double *x)

/************************************************************
 *
 *  exchange_dof():
 *
 *  send/recv appropriate pieces of a node-based double array
 ************************************************************/
{
  COMM_NP_STRUCT *np_base, *np_ptr;
  double *ptr_send_list, *ptr_recv_list;
  register double *ptrd;
  register int *ptr_int, i;
  int p;
  int num_neighbors = dpi->num_neighbors;
  int total_num_send_unknowns;

  if (num_neighbors == 0)
    return;

#ifdef PARALLEL
  total_num_send_unknowns = ptr_node_send[dpi->num_neighbors];
  np_base = alloc_struct_1(COMM_NP_STRUCT, dpi->num_neighbors);
  ptrd = (double *)alloc_dbl_1(total_num_send_unknowns, DBL_NOINIT);
  ptr_send_list = ptrd;

  /*
   * gather up the list of send unknowns
   */
  ptr_int = list_node_send;
  for (i = total_num_send_unknowns; i > 0; i--) {
    *ptrd++ = x[*ptr_int++];
  }

  /*
   * store base address for the start of the entries corresponding
   * to external nodes in this vector
   */
  ptr_recv_list = x + dpi->num_internal_nodes + dpi->num_boundary_nodes;

  np_ptr = np_base;
  for (p = 0; p < dpi->num_neighbors; p++) {
    np_ptr->neighbor_ProcID = cx[p].neighbor_name;
    np_ptr->send_message_buf = (void *)(ptr_send_list + ptr_node_send[p]);
    np_ptr->send_message_length = sizeof(double) * cx[p].num_nodes_send;
    np_ptr->recv_message_buf = (void *)ptr_recv_list;
    np_ptr->recv_message_length = sizeof(double) * cx[p].num_nodes_recv;
    ptr_recv_list += cx[p].num_nodes_recv;
    np_ptr++;
  }
  exchange_neighbor_proc_info(dpi->num_neighbors, np_base);
  safer_free((void **)&np_base);
  safer_free((void **)&ptr_send_list);
#endif
}
/********************************************************************/
/********************************************************************/
/********************************************************************/
