
#include "adapt/resetup_problem.h"

#include <mm_bc.h>
#include <rf_bc.h>
#include <rf_pre_proc.h>
#include <rf_solve.h>
#include <string.h>

#include "dp_map_comm_vec.h"
#include "dp_types.h"
#include "dpi.h"
#include "el_geom.h"
#include "linalg/sparse_matrix.h"
#include "mm_as.h"
#include "mm_as_structs.h"
#include "mm_eh.h"
#include "mm_fill_util.h"
#include "mm_unknown_map.h"
#include "rf_allo.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_io.h"
#include "rf_masks.h"
#include "rf_mp.h"
#include "rf_node_const.h"
#include "rf_solver.h"
#include "rf_util.h"
#include "rf_vars_const.h"
#include "sl_util_structs.h"

int resetup_problem(Exo_DB *exo, /* ptr to the finite element mesh database */
                    Dpi *dpi)    /* distributed processing information */

/********************************************************************
 *
 * setup_problem():
 *
 *      Setup_problem() determines the degrees of freedom at each
 * node and formulates the solution vector. It then determines the
 * communications pattern for exchanging that solution vector between
 * processors.
 *     Lastly, it sets up structures that help to carry out the
 * boundary condition integrals on side sets.
 *
 * NOTE:
 *   This function was formed by taking common parts out of
 **********************************************************************/
{

  pre_process(exo);
  /*
   *  Initialize nodal based structures pertaining to properties
   *  and boundary conditions
   */
  init_nodes(exo, dpi);
  /*
   * Enumerate my own degrees of freedom on this processor.
   */
  setup_local_nodal_vars(exo, dpi);

  /*
   * setup communications patterns between ghost and owned
   * nodes.
   */
  setup_nodal_comm_map(exo, dpi, cx);

  /*
   * Find the global maximum number of unknowns located at any one
   * node on any processor
   */
  MaxVarPerNode = find_MaxUnknownNode();

  /*
   * Exchange my idea of what materials I have at each node with
   * my surrounding processors. Make sure we are all in sync
   */
  setup_external_nodal_matrls(exo, dpi, cx[0]);

  /*
   * Exchange my idea of what degrees of freedom I have with my
   * surrounding processors. Make sure we are all in sync.
   * Owned nodes tell ghost nodes what variables are active
   * at that node.
   */
  setup_external_nodal_vars(exo, dpi, cx);

  /*
   * Finish setting the unknown map on this processor
   */
  set_unknown_map(exo, dpi);

  /*
   * Now determine the communications pattern that is necessary to
   * exchange the solution vector in as efficient a manner as
   * possible
   */
  // log_msg("setup_dof_comm_map...");
  setup_dof_comm_map(exo, dpi, cx);

  /*
   * Output some statistics concerning the communications pattern
   */
  if (Num_Proc > 1)
    output_comm_stats(dpi, cx);

  /*
   * I extracted this from setup_fill_comm_map because some of the
   * renormalization routines make use of them.
   */
  num_fill_unknowns = count_vardofs(FILL, dpi->num_universe_nodes);
  internal_fill_unknowns = count_vardofs(FILL, dpi->num_internal_nodes);
  owned_fill_unknowns = count_vardofs(FILL, (dpi->num_internal_nodes + dpi->num_boundary_nodes));
  boundary_fill_unknowns = owned_fill_unknowns - internal_fill_unknowns;
  external_fill_unknowns = num_fill_unknowns - owned_fill_unknowns;

  /*
   *  Possibly increase the number of variable descriptions to include
   *  those that are not part of the solution vector
   */
  vdesc_augment();

  /*
   *  Setup Boundary condition inter-connectivity
   *  -> Stefan flow bc's need to know what bc's contain the reaction
   *     info.
   *  -> YFLUX bc's need to be connected -> future implementation
   */
  set_up_BC_connectivity();

  /*
   *  Set up the structures necessary to carry out
   *  surface integrals
   */
  //  log_msg("set_up_Surf_BC...");
  set_up_Surf_BC(First_Elem_Side_BC_Array, exo, dpi);

  /*
   *  Set up the Edge boundary condition structures
   */
  //  log_msg("set_up_Edge_BC...");
  set_up_Edge_BC(First_Elem_Edge_BC_Array, exo, dpi);

  /*
   * Set up "boundary" conditions on level set surfaces
   */
  //  set_up_Embedded_BC();

  /* Special 1D
   * Set up surface integral boundary conditions
   * that apply at single nodes.
   */

  //  setup_Point_BC(First_Elem_Side_BC_Array, exo, dpi);

  /*
   *  Print out the edge boudary condition structures
   *  if necessary
   */
  //  if (Debug_Flag) {
  //    for (pg->imtrx = 0; pg->imtrx < upd->Total_Num_Matrices; pg->imtrx++) {
  //      print_setup_Surf_BC(First_Elem_Side_BC_Array[pg->imtrx]);
  //    }
  //  }

  /*
   *  Malloc structures of size Num_Var_Info_Records
   */
  //  for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
  //    ei[pg->imtrx]->VDindex_to_Lvdesc = alloc_int_1(Num_Var_Info_Records, -1);
  //  }
  //  for (int i = 0; i < MAX_ELEMENT_INDICES_RELATED; i++) {
  //    eiRelated[i]->VDindex_to_Lvdesc = alloc_int_1(Num_Var_Info_Records, -1);
  //  }

  /*
   * Setup the storage for temporary quantities of interest that
   * are storred on a "per processor element" basis. These include
   * temporary storage of volumetric quadrature information
   */
  //  setup_element_storage();

  /*
   * Setup some structures for solving problems with shell elements.
   */
  //  init_shell_element_blocks(exo);

  /* Communicate non-shared but needed BC information */
  //  exchange_bc_info();
  Num_Node = exo->num_nodes;

  return 0;
}

int resetup_matrix(struct GomaLinearSolverData **ams, Exo_DB *exo, Dpi *dpi) {
  if ((strcmp(Matrix_Format, "tpetra") == 0) || (strcmp(Matrix_Format, "epetra") == 0)) {
    for (pg->imtrx = 0; pg->imtrx < upd->Total_Num_Matrices; pg->imtrx++) {
      GomaSparseMatrix goma_matrix = ams[pg->imtrx]->GomaMatrixData;
      GomaSparseMatrix_Destroy(&goma_matrix);
      GomaSparseMatrix_CreateFromFormat(&goma_matrix, Matrix_Format);
      ams[pg->imtrx]->GomaMatrixData = goma_matrix;
      int local_nodes = exo->num_nodes;
      GomaSparseMatrix_SetProblemGraph(goma_matrix, num_internal_dofs[pg->imtrx],
                                       num_boundary_dofs[pg->imtrx], num_external_dofs[pg->imtrx],
                                       local_nodes, Nodes, MaxVarPerNode, Matilda, Inter_Mask, exo,
                                       dpi, cx[pg->imtrx], pg->imtrx, Debug_Flag, ams[JAC]);
      ams[pg->imtrx]->solveSetup = 0;
    }
    pg->imtrx = 0;
  } else if (strcmp(Matrix_Format, "msr") == 0) {
    for (pg->imtrx = 0; pg->imtrx < upd->Total_Num_Matrices; pg->imtrx++) {
      int imtrx = pg->imtrx;

      if (ams[imtrx]->DestroySolverData) {
        ams[imtrx]->DestroySolverData(ams[imtrx]);
        ams[imtrx]->DestroySolverData = NULL;
        ams[imtrx]->SolverData = NULL;
      }

      safer_free((void **)&ams[imtrx]->val);
      safer_free((void **)&ams[imtrx]->val_old);
      safer_free((void **)&ams[imtrx]->bindx);
      safer_free((void **)&ams[imtrx]->belfry);
      safer_free((void **)&ams[imtrx]->data_org);

      int *ija = NULL;
      double *a = NULL, *a_old = NULL;
      int *ija_attic = NULL;

      /* Fill=0 means node_to_fill is unused; pass NULL */
      alloc_MSR_sparse_arrays(&ija, &a, &a_old, 0, NULL, exo, dpi);
      alloc_extern_ija_buffer(num_universe_dofs[imtrx],
                              num_internal_dofs[imtrx] + num_boundary_dofs[imtrx], ija, &ija_attic);

      ams[imtrx]->bindx = ija;
      ams[imtrx]->val = a;
      ams[imtrx]->val_old = a_old;
      ams[imtrx]->belfry = ija_attic;
      ams[imtrx]->indx = NULL;
      ams[imtrx]->bpntr = NULL;
      ams[imtrx]->rpntr = NULL;
      ams[imtrx]->cpntr = NULL;

      ams[imtrx]->N_update = num_internal_dofs[imtrx] + num_boundary_dofs[imtrx];
      ams[imtrx]->npn = dpi->num_internal_nodes + dpi->num_boundary_nodes;
      ams[imtrx]->npn_plus =
          dpi->num_internal_nodes + dpi->num_boundary_nodes + dpi->num_external_nodes;
      ams[imtrx]->npu = num_internal_dofs[imtrx] + num_boundary_dofs[imtrx];
      ams[imtrx]->npu_plus = num_universe_dofs[imtrx];
      ams[imtrx]->nnz = ija[num_internal_dofs[imtrx] + num_boundary_dofs[imtrx]] - 1;
      ams[imtrx]->nnz_plus = ija[num_universe_dofs[imtrx]];

#ifdef GOMA_ENABLE_AZTEC
      /* Rebuild data_org — mirrors sl_init() logic; sl_init() cannot be called
       * again due to the static Num_Calls guard in sl_util.c */
      if (Num_Proc == 1) {
        ams[imtrx]->data_org = (int *)array_alloc(1, AZ_COMM_SIZE, sizeof(int));
        ams[imtrx]->data_org[AZ_matrix_type] = AZ_MSR_MATRIX;
        ams[imtrx]->mat_type = AZ_MSR_MATRIX;
        ams[imtrx]->data_org[AZ_N_int_blk] = ams[imtrx]->N_update;
        ams[imtrx]->data_org[AZ_N_bord_blk] = 0;
        ams[imtrx]->data_org[AZ_N_ext_blk] = 0;
        ams[imtrx]->data_org[AZ_N_internal] = ams[imtrx]->N_update;
        ams[imtrx]->data_org[AZ_N_border] = 0;
        ams[imtrx]->data_org[AZ_N_external] = 0;
        ams[imtrx]->data_org[AZ_N_neigh] = 0;
        ams[imtrx]->data_org[AZ_total_send] = 0;
        ams[imtrx]->data_org[AZ_name] = 1 + imtrx;
        ams[imtrx]->data_org[AZ_neighbors] = 0;
        ams[imtrx]->data_org[AZ_rec_length] = 0;
        ams[imtrx]->data_org[AZ_send_length] = 0;
        ams[imtrx]->external = NULL;
        ams[imtrx]->update_index = NULL;
        ams[imtrx]->extern_index = NULL;
      } else {
        int length = AZ_COMM_SIZE + ptr_dof_send[imtrx][dpi->num_neighbors];
        ams[imtrx]->data_org = (int *)smalloc(length * sizeof(int));
        ams[imtrx]->data_org[AZ_N_internal] = num_internal_dofs[imtrx];
        ams[imtrx]->data_org[AZ_N_border] = num_boundary_dofs[imtrx];
        ams[imtrx]->data_org[AZ_N_external] = num_external_dofs[imtrx];
        ams[imtrx]->data_org[AZ_matrix_type] = AZ_MSR_MATRIX;
        ams[imtrx]->mat_type = AZ_MSR_MATRIX;
        ams[imtrx]->data_org[AZ_N_int_blk] = num_internal_dofs[imtrx];
        ams[imtrx]->data_org[AZ_N_bord_blk] = num_boundary_dofs[imtrx];
        ams[imtrx]->data_org[AZ_N_ext_blk] = num_external_dofs[imtrx];
        ams[imtrx]->data_org[AZ_N_neigh] = dpi->num_neighbors;
        ams[imtrx]->data_org[AZ_total_send] = ptr_dof_send[imtrx][dpi->num_neighbors];
        ams[imtrx]->data_org[AZ_name] = 1 + imtrx;
        /* sl_init is always called with cx[0]; cx[0][p] gives per-neighbor info */
        for (int p = 0; p < dpi->num_neighbors; p++) {
          ams[imtrx]->data_org[AZ_neighbors + p] = cx[0][p].neighbor_name;
          ams[imtrx]->data_org[AZ_rec_length + p] = cx[0][p].num_dofs_recv;
          ams[imtrx]->data_org[AZ_send_length + p] = cx[0][p].num_dofs_send;
        }
        for (int i = 0; i < ptr_dof_send[imtrx][dpi->num_neighbors]; i++) {
          ams[imtrx]->data_org[AZ_send_list + i] = list_dof_send[imtrx][i];
        }
      }
#endif /* GOMA_ENABLE_AZTEC */

      ams[imtrx]->solveSetup = 0;
    }
    pg->imtrx = 0;
  } else {
    GOMA_EH(-1, "Unsupported matrix storage format use epetra");
  }
  return 0;
}

// vim: expandtab sw=2 ts=8
