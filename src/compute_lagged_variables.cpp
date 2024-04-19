#include "compute_lagged_variables.h"

extern "C" {
#include "load_field_variables.h"
#include "dp_comm.h"
#include "dpi.h"
#include "el_geom.h"
#include "exo_struct.h"
#include "mm_as.h"
#include "mm_fill_ptrs.h"
#include "mm_fill_util.h"
#include "mm_mp.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_node_const.h"
#include "rf_solve.h"
}

#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>

extern "C" void setup_lagged_variables(Exo_DB *exo, Dpi *dpi, struct Lagged_Variables *lv) {
  if (upd->matrix_index[R_STRESS11] >= 0) {
    // find
    int local_count = 0;
    int local_rows = 0;
    lv->local_node_to_lagged = (int *)malloc(
        (dpi->num_internal_nodes + dpi->num_boundary_nodes + dpi->num_external_nodes) *
        sizeof(int));
    for (int inode = 0;
         inode < (dpi->num_internal_nodes + dpi->num_boundary_nodes + dpi->num_external_nodes);
         inode++) {
      NODAL_VARS_STRUCT *nv = Nodes[inode]->Nodal_Vars_Info[pg->imtrx];
      /*
       * Fill the vector list which points to the unknowns defined at this
       * node...
       */
      int inode_varType[MaxVarPerNode], inode_matID[MaxVarPerNode];
      int row_num_unknowns = fill_variable_vector(inode, inode_varType, inode_matID);
      /*
       * Do a check against the number of unknowns at this
       * node stored in the global array
       */
      if (row_num_unknowns != nv->Num_Unknowns) {
        GOMA_EH(GOMA_ERROR, "Inconsistency counting unknowns.");
      }

      /*
       * Loop over the unknowns defined at this row node
       */
      lv->local_node_to_lagged[inode] = -1;
      for (int iunknown = 0; iunknown < row_num_unknowns; iunknown++) {
        /*
         * Retrieve the var type of the current unknown
         */
        int rowVarType = inode_varType[iunknown];

        if (rowVarType == R_STRESS11) {
          lv->local_node_to_lagged[inode] = local_count;
          local_count++;
          if (inode < (dpi->num_internal_nodes + dpi->num_boundary_nodes)) {
            local_rows++;
          }
        }
      }
    }

    int global_count;
    MPI_Allreduce(&local_rows, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (global_count > 0) {
      lv->global_count = global_count;
      lv->local_count = local_count;
      lv->local_rows = local_rows;
      lv->mapping_index = (int *)malloc(local_count * sizeof(int));
      int RowOffset;
      MPI_Scan(&local_rows, &RowOffset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      RowOffset -= local_rows;
      for (int i = 0; i < local_rows; i++) {
        lv->mapping_index[i] = RowOffset + i;
      }
      int *mapping_index_exchange = (int *)malloc(dpi->num_universe_nodes * sizeof(int));
      for (int i = 0; i < dpi->num_universe_nodes; i++) {
        if (lv->local_node_to_lagged[i] >= 0) {
          mapping_index_exchange[i] = lv->mapping_index[lv->local_node_to_lagged[i]];
        } else {
          mapping_index_exchange[i] = -1;
        }
      }
      exchange_node_int(cx[pg->imtrx], dpi, mapping_index_exchange);
      for (int i = 0; i < dpi->num_universe_nodes; i++) {
        if (lv->local_node_to_lagged[i] >= 0) {
          lv->mapping_index[lv->local_node_to_lagged[i]] = mapping_index_exchange[i];
        }
      }
      free(mapping_index_exchange);

      int v_g[DIM][DIM];
      v_g[0][0] = VELOCITY_GRADIENT11;
      v_g[0][1] = VELOCITY_GRADIENT12;
      v_g[1][0] = VELOCITY_GRADIENT21;
      v_g[1][1] = VELOCITY_GRADIENT22;
      v_g[0][2] = VELOCITY_GRADIENT13;
      v_g[1][2] = VELOCITY_GRADIENT23;
      v_g[2][0] = VELOCITY_GRADIENT31;
      v_g[2][1] = VELOCITY_GRADIENT32;
      v_g[2][2] = VELOCITY_GRADIENT33;

      lv->index = (int *)malloc(MAX_VARIABLE_TYPES * sizeof(int));
      int offset = 0;
      for (int i = V_FIRST; i < V_LAST; i++) {
        if (i >= VELOCITY_GRADIENT11 && i <= VELOCITY_GRADIENT33) {
          lv->index[i] = offset;
          offset++;
        } else {
          lv->index[i] = -1;
        }
      }

      for (int i = 0; i < VIM; i++) {
        for (int j = 0; j < VIM; j++) {
          if (lv->index[v_g[i][j]] >= 0) {
            lv->exchange_lagged[lv->index[v_g[i][j]]] =
                (double *)calloc(1, dpi->num_universe_nodes * sizeof(double));
            lv->lagged_variables[lv->index[v_g[i][j]]] =
                (double *)calloc(1, local_count * sizeof(double));
          }
        }
      }
    }
  }
}

static PetscInt initialize_lagged_matrix(Mat A,
                                         Vec bvec[MAX_PROB_VAR],
                                         Exo_DB *exo,
                                         Dpi *dpi,
                                         dbl *x,
                                         dbl *x_old,
                                         dbl *x_older,
                                         dbl *xdot,
                                         dbl *xdot_old,
                                         struct Lagged_Variables *lv) {
  PetscInt *d_nnz = (PetscInt *)calloc(lv->local_rows, sizeof(PetscInt));
  PetscInt *o_nnz = (PetscInt *)calloc(lv->local_rows, sizeof(PetscInt));
  for (int eb_index = 0; eb_index < exo->num_elem_blocks; eb_index++) {
    int mn = Matilda[eb_index];

    pd = pd_glob[mn];
    cr = cr_glob[mn];
    elc = elc_glob[mn];
    elc_rs = elc_rs_glob[mn];
    gn = gn_glob[mn];
    mp = mp_glob[mn];
    vn = vn_glob[mn];
    evpl = evpl_glob[mn];

    for (int mode = 0; mode < vn->modes; mode++) {
      ve[mode] = ve_glob[mn][mode];
    }

    int e_start = exo->eb_ptr[eb_index];
    int e_end = exo->eb_ptr[eb_index + 1];

    for (int iel = e_start; iel < e_end; iel++) {
      int ielem = iel;

      int err = load_elem_dofptr(ielem, exo, x, x_old, xdot, xdot_old, 0);
      GOMA_EH(err, "load_elem_dofptr");
      err = bf_mp_init(pd);
      GOMA_EH(err, "bf_mp_init");

      int eqn = R_STRESS11;
      for (int i = 0; i < ei[pg->imtrx]->num_local_nodes; i++) {
        int gnn_i = lv->local_node_to_lagged[Proc_Elem_Connect[ei[pg->imtrx]->iconnect_ptr + i]];
        if (gnn_i >= 0) {
          int ldof_i = ei[upd->matrix_index[eqn]]->ln_to_dof[eqn][i];
          for (int j = 0; j < ei[pg->imtrx]->num_local_nodes; j++) {
            int gnn_j =
                lv->local_node_to_lagged[Proc_Elem_Connect[ei[pg->imtrx]->iconnect_ptr + j]];
            int ldof_j = ei[upd->matrix_index[eqn]]->ln_to_dof[eqn][j];
            if (gnn_j >= 0) {
              if (ldof_i >= 0 && ldof_j >= 0 && ((gnn_i < lv->local_rows) || Num_Proc == 1)) {
                if (gnn_i < lv->local_rows) {
                  if (gnn_j >= lv->local_rows) {
                    o_nnz[gnn_i] += 1;
                  } else {
                    d_nnz[gnn_i] += 1;
                  }
                }
              }
            }
          }
        }
      }
    } /* END  for (iel = 0; iel < num_internal_elem; iel++)            */
  }   /* END for (ieb loop) */

  if (Num_Proc == 1) {
    MatSeqAIJSetPreallocation(A, 0, d_nnz);
  } else {
    MatMPIAIJSetPreallocation(A, 0, d_nnz, 0, o_nnz);
  }

  for (int eb_index = 0; eb_index < exo->num_elem_blocks; eb_index++) {
    int mn = Matilda[eb_index];

    pd = pd_glob[mn];
    cr = cr_glob[mn];
    elc = elc_glob[mn];
    elc_rs = elc_rs_glob[mn];
    gn = gn_glob[mn];
    mp = mp_glob[mn];
    vn = vn_glob[mn];
    evpl = evpl_glob[mn];

    for (int mode = 0; mode < vn->modes; mode++) {
      ve[mode] = ve_glob[mn][mode];
    }

    int e_start = exo->eb_ptr[eb_index];
    int e_end = exo->eb_ptr[eb_index + 1];

    for (int iel = e_start; iel < e_end; iel++) {
      int ielem = iel;

      int err = load_elem_dofptr(ielem, exo, x, x_old, xdot, xdot_old, 0);
      GOMA_EH(err, "load_elem_dofptr");
      err = bf_mp_init(pd);
      GOMA_EH(err, "bf_mp_init");
      int ielem_type = ei[pg->imtrx]->ielem_type;
      int ip_total = elem_info(NQUAD, ielem_type); /* number of
                                                    * quadrature pts */

      for (int ip = 0; ip < ip_total; ip++) {
        dbl xi[3];
        dbl s, t, u;

        find_stu(ip, ielem_type, &s, &t, &u);
        xi[0] = s;
        xi[1] = t;
        xi[2] = u;

        /*
         * find quadrature weights for current ip
         */
        dbl wt = Gq_weight(ip, ielem_type);
        fv->wt = wt;

        /*
         * Load up basis function information for ea variable...
         * Old usage: fill_shape
         */
        err = load_basis_functions(xi, bfd);
        GOMA_EH(err, "problem from load_basis_functions");

        err = load_fv();
        GOMA_EH(err, "load_fv");

        /*
         * This has elemental Jacobian transformation and some
         * basic mesh derivatives...
         * Old usage: calc_Jac, jelly_belly
         */
        err = beer_belly();
        GOMA_EH(err, "beer_belly");

        err = load_coordinate_scales(pd->CoordinateSystem, fv);
        GOMA_EH(err, "load_coordinate_scales(fv)");

        err = load_bf_grad();
        GOMA_EH(err, "load_bf_grad");

      err = load_fv_vector();
      GOMA_EH(err, "load_fv_vector");

      err = load_fv_grads();
      GOMA_EH(err, "load_fv_grads");


        int v_g[DIM][DIM];
        v_g[0][0] = VELOCITY_GRADIENT11;
        v_g[0][1] = VELOCITY_GRADIENT12;
        v_g[1][0] = VELOCITY_GRADIENT21;
        v_g[1][1] = VELOCITY_GRADIENT22;
        v_g[0][2] = VELOCITY_GRADIENT13;
        v_g[1][2] = VELOCITY_GRADIENT23;
        v_g[2][0] = VELOCITY_GRADIENT31;
        v_g[2][1] = VELOCITY_GRADIENT32;
        v_g[2][2] = VELOCITY_GRADIENT33;
        int eqn = R_STRESS11;
        for (int i = 0; i < ei[pg->imtrx]->num_local_nodes; i++) {
          int gnn_i = lv->local_node_to_lagged[Proc_Elem_Connect[ei[pg->imtrx]->iconnect_ptr + i]];
          int ldof_i = ei[upd->matrix_index[eqn]]->ln_to_dof[eqn][i];
          if (gnn_i >= 0 && (gnn_i < lv->local_rows || Num_Proc == 1)) {
            for (int a = 0; a < VIM; a++) {
              for (int b = 0; b < VIM; b++) {
                if (lv->index[v_g[a][b]] >= 0) {
                  dbl vec_contrib =
                      fv->grad_v[a][b] * bf[eqn]->phi[ldof_i] * fv->wt * bf[eqn]->detJ;
                  PetscErrorCode err =
                      VecSetValue(bvec[lv->index[v_g[a][b]]], lv->mapping_index[gnn_i],
                                  vec_contrib, ADD_VALUES);
                  CHKERRQ(err);
                }
              }
            }
          }
          for (int j = 0; j < ei[pg->imtrx]->num_local_nodes; j++) {
            int gnn_j =
                lv->local_node_to_lagged[Proc_Elem_Connect[ei[pg->imtrx]->iconnect_ptr + j]];
            int ldof_j = ei[upd->matrix_index[eqn]]->ln_to_dof[eqn][j];
            if (ldof_i >= 0 && ldof_j >= 0 && ((gnn_i < lv->local_rows) || Num_Proc == 1)) {
              dbl mm_contrib = bf[eqn]->phi[ldof_i] * bf[eqn]->phi[ldof_j] * fv->wt * bf[eqn]->detJ;
              PetscInt global_row = lv->mapping_index[gnn_i];
              PetscInt global_col = lv->mapping_index[gnn_j];
              PetscErrorCode err = MatSetValue(A, global_row, global_col, mm_contrib, ADD_VALUES);
              CHKERRQ(err);
            }
          }
        }
      } /* END  for (ip = 0; ip < ip_total; ip++)                      */
    }   /* END  for (iel = 0; iel < num_internal_elem; iel++)            */
  }     /* END for (ieb loop) */

  free(d_nnz);
  free(o_nnz);
  return 0;
}

extern "C" int compute_lagged_variables(Exo_DB *exo,
                                        Dpi *dpi,
                                        dbl *x,
                                        dbl *x_old,
                                        dbl *x_older,
                                        dbl *xdot,
                                        dbl *xdot_old,
                                        struct Lagged_Variables *lv) {
  PetscBool petsc_initialized = PETSC_FALSE;
  PetscInitialized(&petsc_initialized);
  if (!petsc_initialized) {
    PetscErrorCode err = PetscInitializeNoArguments();
    CHKERRQ(err);
  }
  // Setup Petsc solver
  PetscErrorCode ierr;
  Mat A;
  Vec b[MAX_PROB_VAR], sol;
  KSP ksp;
  PC pc;
  ierr = MatCreate(PETSC_COMM_WORLD, &A);

  CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A, "lagged_proj_A");
  CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);
  CHKERRQ(ierr);
  int v_g[DIM][DIM];
  v_g[0][0] = VELOCITY_GRADIENT11;
  v_g[0][1] = VELOCITY_GRADIENT12;
  v_g[1][0] = VELOCITY_GRADIENT21;
  v_g[1][1] = VELOCITY_GRADIENT22;
  v_g[0][2] = VELOCITY_GRADIENT13;
  v_g[1][2] = VELOCITY_GRADIENT23;
  v_g[2][0] = VELOCITY_GRADIENT31;
  v_g[2][1] = VELOCITY_GRADIENT32;
  v_g[2][2] = VELOCITY_GRADIENT33;
  for (int i = 0; i < VIM; i++) {
    for (int j = 0; j < VIM; j++) {
      // if (upd->matrix_index[v_g[i][j]] >= 0) {
        std::string opt = "lagged_proj_b_" + std::to_string(v_g[i][j]);
        ierr = VecCreate(PETSC_COMM_WORLD, &(b[lv->index[v_g[i][j]]]));
        CHKERRQ(ierr);
        ierr =
            VecSetOptionsPrefix(b[lv->index[v_g[i][j]]], opt.c_str());
        CHKERRQ(ierr);
        ierr = VecSetFromOptions(b[lv->index[v_g[i][j]]]);
        CHKERRQ(ierr);
        ierr = VecSetSizes(b[lv->index[v_g[i][j]]], lv->local_rows,
                           lv->global_count);
        CHKERRQ(ierr);
      // }
    }
  }
  MatSetSizes(A, lv->local_rows, lv->local_rows, lv->global_count, lv->global_count);
  initialize_lagged_matrix(A, b, exo, dpi, x, x_old, x_older, xdot, xdot_old, lv);

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);

  for (int i = 0; i < VIM; i++) {
    for (int j = 0; j < VIM; j++) {
      ierr = VecAssemblyBegin(b[lv->index[v_g[i][j]]]);
      CHKERRQ(ierr);
      ierr = VecAssemblyEnd(b[lv->index[v_g[i][j]]]);
      CHKERRQ(ierr);
    }
  }

  ierr = VecCreate(PETSC_COMM_WORLD, &sol);
  CHKERRQ(ierr);
  ierr = VecSetOptionsPrefix(sol, "lagged_proj_sol");
  CHKERRQ(ierr);
  ierr = VecSetFromOptions(sol);
    CHKERRQ(ierr);
  ierr = VecSetSizes(sol, lv->local_rows, lv->global_count);
  CHKERRQ(ierr);

  ierr = KSPCreate(MPI_COMM_WORLD, &ksp);
  CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp, "lagged_proj_ksp");
  CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);
    CHKERRQ(ierr);
  

  ierr = KSPGetPC(ksp, &pc);
  CHKERRQ(ierr);
  ierr = PCSetType(pc, PCBJACOBI);
  CHKERRQ(ierr);
  ierr = PCSetReusePreconditioner(pc, PETSC_TRUE);
  CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPGMRES);
  CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
  CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, A, A);

  for (int i = 0; i < VIM; i++) {
    for (int j = 0; j < VIM; j++) {
      if (lv->index[v_g[i][j]] >= 0) {
        KSPSolve(ksp, b[lv->index[v_g[i][j]]], sol);
        VecDestroy(&(b[lv->index[v_g[i][j]]]));
        VecGetValues(sol, lv->local_rows, lv->mapping_index,
                     lv->lagged_variables[lv->index[v_g[i][j]]]);
        // Exchange the lagged variables
        for (int n = 0; n < dpi->num_universe_nodes; n++) {
          if (lv->local_node_to_lagged[n] >= 0) {
            lv->exchange_lagged[lv->index[v_g[i][j]]][n] =
                lv->lagged_variables[lv->index[v_g[i][j]]]
                                    [lv->local_node_to_lagged[n]];
          }
        }
        exchange_node(cx[pg->imtrx], dpi, lv->exchange_lagged[lv->index[v_g[i][j]]]);
                     
      }
    }
  }
  MatDestroy(&A);
  VecDestroy(&sol);
  KSPDestroy(&ksp);

  return 0;
}