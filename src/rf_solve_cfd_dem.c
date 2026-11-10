/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) 2026 Goma Developers, National Technology & Engineering   *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

/*
 * solve_problem_cfd_dem -- transient Goma solve coupled to LAMMPS DEM.
 *
 * This is the transient path of solve_problem() (rf_solve.c) reduced to what a VANS
 * CFD-DEM run needs, with the coupling calls inserted:
 *
 *   init_vec -> cfd_dem_init (LAMMPS setup, eps^0 from particles) -> x_old = x
 *   loop:
 *     predict x^{n+1}
 *     cfd_dem_apply_void_fraction  (eps DOFs <- particles at t^n, consistent xdot)
 *     solve_nonlinear_problem, time_step_control
 *     accepted -> write output, cfd_dem_advance (drag -> LAMMPS run to t^{n+1}
 *                 -> eps^{n+1}, fluid reaction field), shift history
 *     rejected -> retry with smaller dt (LAMMPS untouched)
 *
 * Features of solve_problem that are not supported here (level set / phase field,
 * augmenting conditions, LIBRARY_MODE, particle tracking, mesh adaptation, XFEM,
 * sensitivities, solid inertia) are rejected at entry.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef GOMA_ENABLE_AZTEC
#include "az_aztec.h"
#endif
#include "brkfix/fix.h"
#include "cfd_dem_coupling.h"
#include "dp_comm.h"
#include "dp_types.h"
#include "dp_utils.h"
#include "dpi.h"
#include "el_elm.h"
#include "el_elm_info.h"
#include "el_geom.h"
#include "el_quality.h"
#include "exo_struct.h"
#include "linalg/sparse_matrix.h"
#include "mm_as.h"
#include "mm_as_structs.h"
#include "mm_bc.h"
#include "mm_eh.h"
#include "mm_fill_ptrs.h"
#include "mm_fill_util.h"
#include "mm_flux.h"
#include "mm_more_utils.h"
#include "mm_mp.h"
#include "mm_mp_structs.h"
#include "mm_post_def.h"
#include "mm_post_proc.h"
#include "mm_sol_nonlinear.h"
#include "mm_unknown_map.h"
#include "mpi.h"
#include "rd_exo.h"
#include "rd_mesh.h"
#include "rf_allo.h"
#include "rf_bc.h"
#include "rf_bc_const.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_io.h"
#include "rf_io_const.h"
#include "rf_io_structs.h"
#include "rf_masks.h"
#include "rf_mp.h"
#include "rf_node_const.h"
#include "rf_solve.h"
#include "rf_solve_segregated.h"
#include "rf_solver.h"
#include "rf_util.h"
#include "sl_matrix_util.h"
#include "sl_petsc.h"
#include "sl_util.h"
#include "sl_util_structs.h"
#include "std.h"
#include "usr_print.h"
#include "wr_exo.h"
#include "wr_soln.h"

#define ROUND_TO_ONE 0.9999999

static void check_supported_features(void) {
  if (TimeIntegration != TRANSIENT)
    GOMA_EH(GOMA_ERROR, "CFD-DEM: Time integration must be transient");
  if (upd->Total_Num_Matrices != 1)
    GOMA_EH(GOMA_ERROR, "CFD-DEM: segregated (multi-matrix) solves are not supported");
  if (nAC > 0)
    GOMA_EH(GOMA_ERROR, "CFD-DEM: augmenting conditions are not supported");
  if (ls != NULL || pfd != NULL)
    GOMA_EH(GOMA_ERROR, "CFD-DEM: level set / phase function problems are not supported");
  if (upd->XFEM)
    GOMA_EH(GOMA_ERROR, "CFD-DEM: XFEM is not supported");
  if (Particle_Dynamics)
    GOMA_EH(GOMA_ERROR, "CFD-DEM: Goma particle tracking cannot be combined with DEM coupling");
  if ((nn_post_fluxes_sens + nn_post_data_sens) > 0)
    GOMA_EH(GOMA_ERROR, "CFD-DEM: flux/data sensitivities are not supported");
  if (tran->ale_adapt)
    GOMA_EH(GOMA_ERROR, "CFD-DEM: mesh adaptation is not supported (fixed mesh is assumed)");
  if (Num_Var_In_Type[pg->imtrx][MESH_DISPLACEMENT1])
    GOMA_EH(GOMA_ERROR, "CFD-DEM: moving meshes are not supported (fixed mesh is assumed)");
  if (Num_Var_In_Type[pg->imtrx][VOID_FRACTION] == 0)
    GOMA_EH(GOMA_ERROR, "CFD-DEM: needs the VOID_FRACTION unknown (EQ = void_fraction ...)");
#ifdef LIBRARY_MODE
  GOMA_EH(GOMA_ERROR, "CFD-DEM: not available in LIBRARY_MODE");
#endif
}

void solve_problem_cfd_dem(Exo_DB *exo, /* ptr to the finite element mesh database  */
                           Dpi *dpi)    /* distributed processing information       */
{
  int *ija = NULL;      /* column pointer array (MSR) */
  double *a = NULL;     /* nonzero array (MSR) */
  double *a_old = NULL; /* nonzero array (MSR) */
  int *ija_attic = NULL;

  double *x = NULL, *x_save = NULL, *x_pred = NULL;
  double *x_old = NULL, *x_older = NULL, *x_oldest = NULL;
  double *xdot = NULL, *xdot_save = NULL, *xdot_old = NULL, *xdot_older = NULL;
  double *x_update = NULL, *resid_vector = NULL, *resid_vector_sens = NULL, *scale = NULL;
  double **x_sens_p = NULL;
  int *node_to_fill = NULL;

  static struct GomaLinearSolverData *ams[NUM_ALSS] = {NULL};
  struct Results_Description *rd = NULL;
  double *gvec = NULL;
  double ***gvec_elem = NULL;
  double *gv = NULL;
  double *x_pp = NULL;
  int *gindex = NULL, gsize = 0;
  FILE *file = NULL;

  char tspstring[MAX_FNL];
  int n, nt = 0, time_step_reform;
  int converged = TRUE, success_dt = TRUE, failed_recently_countdown = 0;
  int i, num_total_nodes, numProcUnknowns;
  int const_delta_t, const_delta_ts, step_print;
  int step_fix = 0, i_fix = 0;
  int good_mesh = TRUE;
  int nprint = 0;
  int tnv, tev, tnv_post, tev_post;
  int error, err;
  unsigned int matrix_systems_mask = 1;
  double time_print, theta = 0.0, time, time1 = 0.0;
  double delta_t, delta_t_new = 0.0;
  double delta_t_old, delta_t_older, delta_t_oldest = 0.0;
  double timeValueRead = 0.0, timeValueReadTrans = 0.0;
  double eps;
  int i_print;
  int w;

  static const char yo[] = "solve_problem_cfd_dem";

  check_supported_features();

  if (Num_Proc > 1 && tran->fix_freq > 0) {
    step_fix = 1; /* Always fix on the first timestep to match print frequency */
  }
  tran->time_value = time1;

  if (Unlimited_Output && strlen(Soln_OutFile)) {
    file = fopen(Soln_OutFile, "w");
    if (file == NULL)
      GOMA_EH(GOMA_ERROR, "%s: can not open solution file %s for writing", yo, Soln_OutFile);
  }

  /*
   * Results description, exodus output setup (as in solve_problem)
   */
  tnv = cnt_nodal_vars();
  tev = cnt_elem_vars(exo);
  if (tnv < 0 || tev < 0)
    GOMA_EH(GOMA_ERROR, "%s: bad tnv/tev", yo);

  rd = alloc_struct_1(struct Results_Description, 1);
  rd->nev = 0;
  rd->nhv = 0;
  rd->ngv = 6;
  error = load_global_var_info(rd, 0, "CONV");
  error = load_global_var_info(rd, 1, "NEWT_IT");
  error = load_global_var_info(rd, 2, "MAX_IT");
  error = load_global_var_info(rd, 3, "CONVORDER");
  error = load_global_var_info(rd, 4, "CONVRATE");
  error = load_global_var_info(rd, 5, "MESH_VOLUME");

  Spec_source_inventory = Dmatrix_birth(upd->Num_Mat, upd->Max_Num_Species_Eqn + 1);
  Spec_source_lumped_mass = alloc_dbl_1(exo->num_nodes, 0.0);

  gv = alloc_dbl_1(rd->ngv, 0.0);

  error = load_nodal_tkn(rd, &tnv, &tnv_post);
  GOMA_EH(error, "%s: problem with load_nodal_tkn()", yo);
  if (Num_Export_XP > 0 && POROUS_SATURATION != -1) {
    asdv(&x_pp, Num_Export_XS * exo->num_nodes);
  }
  error = load_elem_tkn(rd, exo, tev, &tev_post);
  GOMA_EH(error, "%s: problem with load_elem_tkn()", yo);
  check_parallel_error("Results file error");

  gvec_elem = (double ***)alloc_ptr_1(exo->num_elem_blocks);
  if ((tev + tev_post) > 0) {
    for (i = 0; i < exo->num_elem_blocks; i++) {
      gvec_elem[i] = (double **)alloc_ptr_1(tev + tev_post);
    }
  }
  wr_result_prelim_exo(rd, exo, ExoFileOut, gvec_elem);
  asdv(&gvec, Num_Node);

  /*
   * Solution vectors and linear solver structures
   */
  num_total_nodes = dpi->num_universe_nodes;
  numProcUnknowns = NumUnknowns[pg->imtrx] + NumExtUnknowns[pg->imtrx];

  asdv(&resid_vector, numProcUnknowns);
  asdv(&resid_vector_sens, numProcUnknowns);
  asdv(&scale, numProcUnknowns);

  for (i = 0; i < NUM_ALSS; i++) {
    ams[i] = alloc_struct_1(struct GomaLinearSolverData, 1);
  }
#ifdef GOMA_ENABLE_AZTEC
  AZ_set_proc_config(ams[0]->proc_config, MPI_COMM_WORLD);
#endif

  x_save = alloc_dbl_1(numProcUnknowns, 0.0);
  x = alloc_dbl_1(numProcUnknowns, 0.0);
  x_old = alloc_dbl_1(numProcUnknowns, 0.0);
  x_older = alloc_dbl_1(numProcUnknowns, 0.0);
  x_oldest = alloc_dbl_1(numProcUnknowns, 0.0);
  xdot_save = alloc_dbl_1(numProcUnknowns, 0.0);
  xdot = alloc_dbl_1(numProcUnknowns, 0.0);
  xdot_old = alloc_dbl_1(numProcUnknowns, 0.0);
  xdot_older = alloc_dbl_1(numProcUnknowns, 0.0);
  x_update = alloc_dbl_1(numProcUnknowns + numProcUnknowns, 0.0);

  set_solid_inertia();
  if (tran->solid_inertia)
    GOMA_EH(GOMA_ERROR, "CFD-DEM: solid inertia (Newmark) is not supported");

  node_to_fill = alloc_int_1(num_total_nodes, 0);

  pg->matrices = malloc(sizeof(struct Matrix_Data));
  pg->matrices[pg->imtrx].ams = ams[JAC];
  pg->matrices[pg->imtrx].x = x;
  pg->matrices[pg->imtrx].x_old = x_old;
  pg->matrices[pg->imtrx].x_older = x_older;
  pg->matrices[pg->imtrx].xdot = xdot;
  pg->matrices[pg->imtrx].xdot_old = xdot_old;
  pg->matrices[pg->imtrx].x_update = x_update;
  pg->matrices[pg->imtrx].scale = scale;
  pg->matrices[pg->imtrx].resid_vector = resid_vector;

  /* Allocate sparse matrix (same formats as solve_problem) */
  ams[JAC]->GomaMatrixData = NULL;
  if ((strcmp(Matrix_Format, "tpetra") == 0) || (strcmp(Matrix_Format, "epetra") == 0)) {
    err = check_compatible_solver();
    GOMA_EH(err, "Incompatible matrix solver for tpetra, tpetra supports stratimikos");
    check_parallel_error("Matrix format / Solver incompatibility");
    GomaSparseMatrix goma_matrix;
    goma_error gerr = GomaSparseMatrix_CreateFromFormat(&goma_matrix, Matrix_Format);
    GOMA_EH(gerr, "GomaSparseMatrix_CreateFromFormat");
    int local_nodes = Num_Internal_Nodes + Num_Border_Nodes + Num_External_Nodes;
    gerr = GomaSparseMatrix_SetProblemGraph(
        goma_matrix, num_internal_dofs[pg->imtrx], num_boundary_dofs[pg->imtrx],
        num_external_dofs[pg->imtrx], local_nodes, Nodes, MaxVarPerNode, Matilda, Inter_Mask, exo,
        dpi, cx[pg->imtrx], pg->imtrx, Debug_Flag, ams[JAC]);
    GOMA_EH(gerr, "GomaSparseMatrix_SetProblemGraph");
    ams[JAC]->GomaMatrixData = goma_matrix;
#if defined(GOMA_ENABLE_PETSC) && !(PETSC_USE_COMPLEX)
  } else if (strcmp(Matrix_Format, "petsc") == 0) {
    err = check_compatible_solver();
    GOMA_EH(err, "Incompatible matrix solver for petsc, solver must be petsc");
    check_parallel_error("Matrix format / Solver incompatibility");
    pg->imtrx = 0;
    goma_error gerr = goma_setup_petsc_matrix(
        ams[JAC], exo, dpi, x, x_old, xdot, xdot_old, num_internal_dofs[pg->imtrx],
        num_boundary_dofs[pg->imtrx], num_external_dofs[pg->imtrx], pg->imtrx);
    GOMA_EH(gerr, "goma_setup_petsc_matrix");
#endif
  } else if (strcmp(Matrix_Format, "msr") == 0) {
    log_msg("alloc_MSR_sparse_arrays...");
    alloc_MSR_sparse_arrays(&ija, &a, &a_old, 0, node_to_fill, exo, dpi);
    alloc_extern_ija_buffer(num_universe_dofs[pg->imtrx],
                            num_internal_dofs[pg->imtrx] + num_boundary_dofs[pg->imtrx], ija,
                            &ija_attic);
    ams[JAC]->bindx = ija;
    ams[JAC]->val = a;
    ams[JAC]->belfry = ija_attic;
    ams[JAC]->val_old = a_old;
    ams[JAC]->indx = NULL;
    ams[JAC]->bpntr = NULL;
    ams[JAC]->rpntr = NULL;
    ams[JAC]->cpntr = NULL;
    ams[JAC]->npn = dpi->num_internal_nodes + dpi->num_boundary_nodes;
    ams[JAC]->npn_plus =
        dpi->num_internal_nodes + dpi->num_boundary_nodes + dpi->num_external_nodes;
    ams[JAC]->npu = num_internal_dofs[pg->imtrx] + num_boundary_dofs[pg->imtrx];
    ams[JAC]->npu_plus = num_universe_dofs[pg->imtrx];
    ams[JAC]->nnz = ija[num_internal_dofs[pg->imtrx] + num_boundary_dofs[pg->imtrx]] - 1;
    ams[JAC]->nnz_plus = ija[num_universe_dofs[pg->imtrx]];
  } else if (strcmp(Matrix_Format, "vbr") == 0) {
    log_msg("alloc_VBR_sparse_arrays...");
    alloc_VBR_sparse_arrays(ams[JAC], exo, dpi);
    ija_attic = NULL;
    ams[JAC]->belfry = ija_attic;
    a = ams[JAC]->val;
    if (!save_old_A)
      a_old = ams[JAC]->val_old = NULL;
  } else {
    GOMA_EH(GOMA_ERROR, "CFD-DEM: unsupported sparse matrix format: %s", Matrix_Format);
  }
#if defined(GOMA_ENABLE_PETSC) && !(PETSC_USE_COMPLEX)
  if (upd->petsc_solve_post_proc && rd->TotalNVPostOutput) {
    goma_setup_petsc_post_proc_matrix(exo, dpi, x, x_old, xdot, xdot_old);
  }
#endif

  /*
   * Initial condition
   */
  init_vec(x, cx[0], exo, dpi, NULL, 0, &timeValueRead);
#ifndef ALLOW_NEGATIVE_TIMES_PLEASE
  if (tran->init_time < 0.0) {
    tran->init_time = timeValueRead;
    DPRINTF(stdout, "\n Initial Simulation Time Has been set to %g\n", timeValueRead);
  }
#endif
  if (Conformation_Flag == 1)
    initial_guess_stress_to_log_conf(x, num_total_nodes);

  if (Output_Variable_Stats) {
    err = variable_stats(x, timeValueRead, FALSE);
    GOMA_EH(err, "Problem with variable_stats!");
  }

  /*
   * Transient setup
   */
  double delta_t0 = tran->Delta_t0;
  double delta_t_min = tran->Delta_t_min;
  double delta_t_max = tran->Delta_t_max;
  double max_time_steps = tran->MaxTimeSteps;
  double time_max = tran->TimeMax;
  eps = tran->eps;

  if (delta_t0 < 0.0) {
    delta_t0 = -delta_t0;
    const_delta_t = 1;
  } else {
    const_delta_t = 0;
  }

  time = time1 = tran->init_time;
  tran->time_value = tran->time_value_old = time1;
  tran->delta_t_old = delta_t0;
  if (delta_t0 > delta_t_max)
    delta_t0 = delta_t_max;
  delta_t = delta_t_old = delta_t_older = delta_t0;
  tran->delta_t = delta_t;
  tran->delta_t_avg = delta_t;

  x_pred = alloc_dbl_1(numProcUnknowns, 0.0);
  x_pred_static = x_pred;

  find_and_set_Dirichlet(x, xdot, exo, dpi);
  exchange_dof(cx[0], dpi, x, 0);

  /*
   * Start LAMMPS and replace the initial void fraction with the particle-based one.
   * xdot = 0 initially, consistent with x_old = x below.
   */
  err = cfd_dem_init(exo, dpi, x, time);
  GOMA_EH(err, "cfd_dem_init");
  err = cfd_dem_apply_void_fraction(x, x, xdot, xdot_old, 0.0, 0.0);
  GOMA_EH(err, "cfd_dem_apply_void_fraction");
  exchange_dof(cx[0], dpi, x, 0);

  dcopy1(numProcUnknowns, x, x_old);
  dcopy1(numProcUnknowns, x_old, x_older);
  dcopy1(numProcUnknowns, x_older, x_oldest);

  time_print = time;
  step_print = 1;
  af->Sat_hyst_reevaluate = FALSE;

  sl_init(matrix_systems_mask, ams, exo, dpi, cx[0]);
  check_parallel_error("Aztec Initialization");

  nt = 0;
  time_step_reform = Time_Jacobian_Reformation_stride;
  const_delta_ts = const_delta_t;

  if (Write_Initial_Solution) {
    if (file != NULL) {
      error = write_ascii_soln(x, resid_vector, numProcUnknowns, NULL, 0, time, file);
      if (error != 0)
        DPRINTF(stderr, "%s:  error writing ASCII soln file\n", yo);
    }
    (void)write_solution(ExoFileOut, resid_vector, x, x_sens_p, x_old, xdot, xdot_old, tev,
                         tev_post, gv, rd, gvec, gvec_elem, &nprint, delta_t, theta, time, x_pp,
                         exo, dpi);
    nprint++;
  }

  if (nEQM > 0) {
    DPRINTF(stdout, "\nINITIAL ELEMENT QUALITY CHECK---\n");
    good_mesh = element_quality(exo, x, ams[0]->proc_config);
  }

  /*******************************************************************
   *  TIME STEP LOOP
   *******************************************************************/
  for (n = 0; n < max_time_steps; n++) {
    time1 = time + delta_t;
    if (time1 > time_max) {
      DPRINTF(stdout, "\t\tLAST TIME STEP!\n");
      time1 = time_max;
      delta_t = time1 - time;
      tran->delta_t = delta_t;
      tran->delta_t_avg = 0.25 * (delta_t + delta_t_old + delta_t_older + delta_t_oldest);
    }
    tran->time_value = time1;

    /* time dependent external fields (the DEM drag fields are not file based) */
    if (efv->ev) {
      timeValueReadTrans = time;
      for (w = 0; w < efv->Num_external_field; w++) {
        if (strcmp(efv->field_type[w], "transient") == 0) {
          err = rd_trans_vectors_from_exoII(x_old, efv->file_nm[w], w, n, &timeValueReadTrans,
                                            exo, cx[0], dpi);
          if (err != 0)
            DPRINTF(stderr, "%s: err from rd_trans_vectors_from_exoII\n", yo);
        }
      }
    }

    /* Backward Euler start-up, then the user's scheme */
    if (nt == 0) {
      theta = 0.0;
      const_delta_t = 1;
    } else if (nt >= 3) {
      theta = tran->theta;
      const_delta_t = const_delta_ts;
    }

    nullify_dirichlet_bcs();
    find_and_set_Dirichlet(x, xdot, exo, dpi);

    if (ProcID == 0) {
      if (theta == 0.0)
        strcpy(tspstring, "(BE)");
      else if (theta == 0.5)
        strcpy(tspstring, "(CN)");
      else if (theta == 1.0)
        strcpy(tspstring, "(FE)");
      else
        sprintf(tspstring, "(TSP %3.1f)", theta);
      fprintf(stdout, "\n=> Try for soln at t=%g with dt=%g [%d for %d] %s\n", time1, delta_t, nt,
              n, tspstring);
      log_msg("Predicting try at t=%g, dt=%g [%d for %d so far] %s", time1, delta_t, nt, n,
              tspstring);
    }

    predict_solution(numProcUnknowns, delta_t, delta_t_old, delta_t_older, theta, x, x_old,
                     x_older, x_oldest, xdot, xdot_old, xdot_older);
    exchange_dof(cx[0], dpi, x, 0);
    exchange_dof(cx[0], dpi, xdot, 0);

    find_and_set_Dirichlet(x, xdot, exo, dpi);

    /* particle void fraction into the (FIX'd) VOID_FRACTION dofs, with consistent xdot */
    err = cfd_dem_apply_void_fraction(x, x_old, xdot, xdot_old, delta_t, theta);
    GOMA_EH(err, "cfd_dem_apply_void_fraction");

    exchange_dof(cx[0], dpi, x, 0);
    exchange_dof(cx[0], dpi, xdot, 0);

    dcopy1(numProcUnknowns, x, x_pred);

    dcopy1(numProcUnknowns, x, x_save);
    dcopy1(numProcUnknowns, xdot, xdot_save);
    tran->current_theta = theta;
    err = solve_nonlinear_problem(ams[JAC], x, delta_t, theta, x_old, x_older, xdot, xdot_old,
                                  resid_vector, x_update, scale, &converged, &nprint, tev,
                                  tev_post, gv, rd, gindex, &gsize, gvec, gvec_elem, time1, exo,
                                  dpi, cx[0], n, &time_step_reform, FALSE, NULL, NULL, time1,
                                  resid_vector_sens, NULL, x_sens_p, NULL);
    if (err == -1)
      converged = FALSE;
    evpl_glob[0]->update_flag = 0;
    af->Sat_hyst_reevaluate = FALSE;

    exchange_dof(cx[0], dpi, x, 0);
    exchange_dof(cx[0], dpi, xdot, 0);

    if (!converged) {
      dcopy1(numProcUnknowns, x_save, x);
      dcopy1(numProcUnknowns, xdot_save, xdot);
    }

    good_mesh = element_quality(exo, x, ams[0]->proc_config);

    if (converged) {
      delta_t_new = time_step_control(delta_t, delta_t_old, const_delta_t, x, x_pred, x_old, NULL,
                                      NULL, eps, &success_dt, tran->use_var_norm);
      if (const_delta_t) {
        success_dt = TRUE;
        delta_t_new = delta_t;
      } else if (failed_recently_countdown > 0) {
        delta_t_new = delta_t;
        failed_recently_countdown--;
      } else if (delta_t_new > delta_t_max) {
        delta_t_new = delta_t_max;
      } else if (delta_t_new < tran->resolved_delta_t_min) {
        delta_t_new = tran->resolved_delta_t_min;
        success_dt = TRUE;
        DPRINTF(stderr, "\n\tminimum resolved step limit! - step control\n");
      }
    }

    if (converged && success_dt) {
      const double time_prev = time;
      nt += 1;
      time = time1;

      /* printing decision */
      i_print = 0;
      i_fix = 0;
      if (tran->print_freq == 0) {
        if ((time > time_print) || (fabs(time - time_print) < (1.e-4 * tran->print_delt))) {
          i_print = 1;
          i_fix = 1;
          if (tran->print_delt2 < 0. || time < tran->print_delt2_time)
            time_print += tran->print_delt;
          else
            time_print += tran->print_delt2;
        }
      } else if (nt == step_print) {
        i_print = 1;
        step_print += tran->print_freq;
      }
      if (time1 >= (ROUND_TO_ONE * time_max))
        i_print = 1;

      err = usr_print(&time, delta_t, x, NULL, -1);
      GOMA_EH(err, "usr_print");

      if (i_print) {
        if (file != NULL) {
          error = write_ascii_soln(x, resid_vector, numProcUnknowns, NULL, 0, time, file);
          if (error != 0)
            fprintf(stderr, "%s:  error writing ASCII soln file\n", yo);
        }
        if (Write_Intermediate_Solutions == 0) {
          write_solution(ExoFileOut, resid_vector, x, x_sens_p, x_old, xdot, xdot_old, tev,
                         tev_post, gv, rd, gvec, gvec_elem, &nprint, delta_t, theta, time, x_pp,
                         exo, dpi);
          nprint++;
        }
      }
      evpl_glob[0]->update_flag = 1;

#ifdef PARALLEL
      if ((step_fix != 0 && nt == step_fix) || ((i_fix == 1) && (tran->fix_freq > 0))) {
        MPI_Barrier(MPI_COMM_WORLD);
        fix_output();
        step_fix += tran->fix_freq * tran->print_freq;
      }
#endif

      /*
       * DEM half of the step: drag from the converged fluid at t^{n+1}, LAMMPS advances
       * the particles to t^{n+1}, then eps and the fluid reaction are rebuilt.
       */
      err = cfd_dem_advance(x, x_old, xdot, xdot_old, time_prev, time, exo, dpi);
      GOMA_EH(err, "cfd_dem_advance");

      /* keep the printing times on the grid */
      if (tran->print_freq == 0 && success_dt) {
        if ((time + 1.2 * delta_t_new >= time_print) && (time_print > time)) {
          delta_t_new = time_print - time;
          DPRINTF(stdout, "reset delta_t = %g to maintain printing frequency\n", delta_t_new);
          if (delta_t_new <= 0)
            GOMA_EH(GOMA_ERROR, "error with time-step printing control");
        } else if (time >= time_print) {
          if (delta_t_new != tran->print_delt) {
            delta_t_new = tran->print_delt;
            DPRINTF(stdout, "reset delta_t = %g to maintain printing frequency\n", delta_t_new);
            if (delta_t_new <= 0)
              GOMA_EH(GOMA_ERROR, "error with time-step printing control");
          }
        }
      }

      /* history shift */
      dcopy1(numProcUnknowns, xdot_old, xdot_older);
      dcopy1(numProcUnknowns, xdot, xdot_old);
      dcopy1(numProcUnknowns, x_older, x_oldest);
      dcopy1(numProcUnknowns, x_old, x_older);
      dcopy1(numProcUnknowns, x, x_old);
      delta_t_oldest = delta_t_older;
      delta_t_older = delta_t_old;
      delta_t_old = delta_t;
      tran->delta_t_old = delta_t_old;
      tran->time_value_old = time;
      delta_t = delta_t_new;
      tran->delta_t = delta_t;
      tran->delta_t_avg = 0.25 * (delta_t + delta_t_old + delta_t_older + delta_t_oldest);

      for (i = 0; i < nn_post_fluxes; i++) {
        (void)evaluate_flux(exo, dpi, pp_fluxes[i]->ss_id, pp_fluxes[i]->flux_type,
                            pp_fluxes[i]->flux_type_name, pp_fluxes[i]->blk_id,
                            pp_fluxes[i]->species_number, pp_fluxes[i]->flux_filenm,
                            pp_fluxes[i]->profile_flag, x, xdot, NULL, delta_t_old, time, 1);
      }
      for (i = 0; i < nn_volume; i++) {
        evaluate_volume_integral(exo, dpi, pp_volume[i]->volume_type, pp_volume[i]->volume_name,
                                 pp_volume[i]->blk_id, pp_volume[i]->species_no,
                                 pp_volume[i]->volume_fname, pp_volume[i]->params,
                                 pp_volume[i]->num_params, NULL, x, xdot, delta_t_old, time, 1);
      }
      if (Output_Variable_Stats) {
        err = variable_stats(x, time, (time >= time_max) ? Output_Variable_Regression : FALSE);
        GOMA_EH(err, "Problem with variable_stats!");
      }

      /* march to steady state (fluid and particles both settled) */
      if (tran->march_to_steady_state && n > 0) {
        double distance = vector_distance(NumUnknowns[0], x, x_older);
        P0PRINTF("\nDelta x %g\n", distance);
        if (distance <= tran->steady_state_tolerance) {
          P0PRINTF("\n Steady state reached \n");
          goto free_and_clear;
        }
      }

      if (time1 >= (ROUND_TO_ONE * time_max)) {
        DPRINTF(stdout, "\t\tout of time!\n");
        goto free_and_clear;
      }
      if (!good_mesh)
        goto free_and_clear;

    } else {
      /* rejected step: LAMMPS has not moved, just retry with a smaller dt */
      if (converged && delta_t < tran->resolved_delta_t_min / tran->time_step_decelerator) {
        DPRINTF(stderr, "\n\tminimum resolved step limit! - not converged\n");
        delta_t_oldest = delta_t_older;
        delta_t_older = delta_t_old;
        delta_t_old = delta_t;
        tran->delta_t_old = delta_t_old;
        tran->time_value_old = time;
        delta_t = tran->resolved_delta_t_min;
      } else {
        DPRINTF(stdout, "\n\tlast time step failed, dt *= %g for next try!\n",
                tran->time_step_decelerator);
        delta_t *= tran->time_step_decelerator;
        evpl_glob[0]->update_flag = 2;
        af->Sat_hyst_reevaluate = 0;
        failed_recently_countdown = tran->const_dt_after_failure;
      }
      tran->delta_t = delta_t;
      tran->delta_t_avg = 0.25 * (delta_t + delta_t_old + delta_t_older + delta_t_oldest);
      time1 = time + delta_t;
      tran->time_value = time1;
    }

    if (delta_t <= delta_t_min) {
      DPRINTF(stderr, "\n\tdelta_t = %e < %e\n\n", delta_t, delta_t_min);
      DPRINTF(stdout, "time step too small, I'm giving up!\n");
      break;
    }
  } /* end of time step loop */

free_and_clear:

  cfd_dem_finalize();

  for (i = 0; i < Num_ROT; i++) {
    safer_free((void **)&(ROT_Types[i].elems));
  }
  safer_free((void **)&ROT_Types);
  safer_free((void **)&node_to_fill);
  safer_free((void **)&resid_vector);
  safer_free((void **)&resid_vector_sens);
  safer_free((void **)&scale);
  safer_free((void **)&x);
  safer_free((void **)&gv);
  safer_free((void **)&x_pp);
  safer_free((void **)&x_pred);
  x_pred_static = NULL;
  safer_free((void **)&x_save);
  safer_free((void **)&xdot_save);
  safer_free((void **)&x_old);
  safer_free((void **)&x_older);
  safer_free((void **)&x_oldest);
  safer_free((void **)&xdot);
  safer_free((void **)&xdot_old);
  safer_free((void **)&xdot_older);
  safer_free((void **)&x_update);

  Dmatrix_death(Spec_source_inventory, upd->Num_Mat, upd->Max_Num_Species_Eqn + 1);
  safer_free((void **)&Spec_source_lumped_mass);

  for (i = 0; i < MAX_NUMBER_MATLS; i++) {
    for (int m = 0; m < MAX_MODES; m++) {
      safer_free((void **)&(ve_glob[i][m]->gn));
      safer_free((void **)&(ve_glob[i][m]->time_const_st));
      safer_free((void **)&(ve_glob[i][m]));
    }
    safer_free((void **)&(vn_glob[i]));
  }
  safer_free((void **)&ve);

  if (num_universe_dofs[pg->imtrx] !=
      (num_internal_dofs[pg->imtrx] + num_boundary_dofs[pg->imtrx])) {
    safer_free((void **)&ija_attic);
  }

  {
    GomaSparseMatrix goma_matrix = ams[JAC]->GomaMatrixData;
    GomaSparseMatrix_Destroy(&goma_matrix);
  }
#ifdef GOMA_ENABLE_PETSC
  if (strcmp(Matrix_Format, "petsc") == 0) {
    err = goma_petsc_free_matrix(ams[JAC]);
    GOMA_EH(err, "free petsc matrix");
  }
#endif
  sl_free(matrix_systems_mask, ams);
  for (i = 0; i < NUM_ALSS; i++) {
    safer_free((void **)&(ams[i]));
  }

  safer_free((void **)&gvec);
  i = 0;
  for (int eb_indx = 0; eb_indx < exo->num_elem_blocks; eb_indx++) {
    for (int ev_indx = 0; ev_indx < rd->nev; ev_indx++) {
      if (exo->elem_var_tab[i++] == 1) {
        safer_free((void **)&(gvec_elem[eb_indx][ev_indx]));
      }
    }
    safer_free((void **)&(gvec_elem[eb_indx]));
  }
  safer_free((void **)&gvec_elem);
  safer_free((void **)&rd);
  for (i = 0; i < num_total_nodes; i++) {
    free(Local_Offset[0][i]);
    free(Dolphin[0][i]);
  }
  free(Dolphin[0]);
  free(Local_Offset[0]);
  safer_free((void **)&Local_Offset);
  safer_free((void **)&Dolphin);

  if (file != NULL)
    fclose(file);
}
