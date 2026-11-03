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

/* directs and controls solution process for hunting with zero and
 * first order  hunting
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GOMA_AC_HUNT_C
#include "ac_hunt.h"
#include "ac_update_parameter.h"
#ifdef GOMA_ENABLE_AZTEC
#include "az_aztec.h"
#endif
#include "brkfix/fix.h"
#include "decomp_interface.h"
#include "dp_comm.h"
#include "dp_types.h"
#include "dp_utils.h"
#include "dpi.h"
#include "el_geom.h"
#include "el_quality.h"
#include "exo_struct.h"
#include "linalg/sparse_matrix.h"
#include "mm_as.h"
#include "mm_as_structs.h"
#include "mm_augc_util.h"
#include "mm_bc.h"
#include "mm_eh.h"
#include "mm_fill_util.h"
#include "mm_flux.h"
#include "mm_more_utils.h"
#include "mm_mp.h"
#include "mm_mp_const.h"
#include "mm_mp_structs.h"
#include "mm_post_def.h"
#include "mm_post_proc.h"
#include "mm_sol_nonlinear.h"
#include "mpi.h"
#include "rf_allo.h"
#include "rf_bc.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_io.h"
#include "rf_io_const.h"
#include "rf_io_structs.h"
#include "rf_masks.h"
#include "rf_mp.h"
#include "rf_node_const.h"
#include "rf_solve.h"
#include "rf_solver.h"
#include "rf_util.h"
#include "sl_auxutil.h"
#include "sl_matrix_util.h"
#ifdef GOMA_ENABLE_PETSC
#include "sl_petsc.h"
#include "sl_petsc_complex.h"
#endif
#include "sl_util.h" /* defines sl_init() */
#include "sl_util_structs.h"
#include "std.h"
#include "wr_exo.h"
#include "wr_soln.h"

/*

   HUNTING WITH ZEROTH AND FIRST ORDER CONTINUATION
   WITH AUGMENTING CONDITIONS

   BASED ON solve_problem() IN rf_solve.c

   BY IAN GATES & ROBERT SECOR

   2/98 - 11/99

*/

void hunt_problem(Comm_Ex *cx, /* array of communications structures */
                  Exo_DB *exo, /* ptr to the finite element mesh database */
                  Dpi *dpi)    /* distributed processing information */
{
  int *ija = NULL;      /* column pointer array                         */
  double *a = NULL;     /* nonzero array                                */
  double *a_old = NULL; /* nonzero array                                */
  double *x = NULL;     /* solution vector                              */

  int iAC;                 /* COUNTER                                      */
  double *x_AC = NULL;     /* SOLUTION VECTOR OF EXTRA UNKNOWNS            */
  double *x_AC_old = NULL; /* old SOLUTION VECTOR OF EXTRA UNKNOWNS        */
  double *x_AC_dot = NULL;

  int iHC; /* COUNTER                                      */

  int *ija_attic = NULL; /* storage for external dofs                    */

  int eb_indx, ev_indx;

  /*
   * variables for path traversal
   */

  double *x_old = NULL;    /* old solution vector                          */
  double *x_older = NULL;  /* older solution vector                        */
  double *x_oldest = NULL; /* oldest solution vector saved                 */
  double *xdot = NULL;     /* current path derivative of soln              */
  double *xdot_old = NULL;
  double *x_update = NULL;

  double *x_sens = NULL;    /* solution sensitivity */
  double **x_sens_p = NULL; /* solution sensitivity for parameters */
  int num_pvector = 0;      /*  number of solution sensitivity vectors */
  struct GomaLinearSolverData *ams[NUM_ALSS] = {NULL};
  /* sl_util_structs.h */

  double *resid_vector = NULL;      /* residual */
  double *resid_vector_sens = NULL; /* residual sensitivity */
  double *scale = NULL;             /* scale vector for modified newton */

  int *node_to_fill = NULL;

  int n;                /* total number of path steps attempted */
  int ni;               /* total number of nonlinear solves */
  int nt;               /* total number of successful path steps */
  int path_step_reform; /* counter for jacobian reformation stride */
  int converged;        /* success or failure of Newton iteration */
  int success_ds;       /* success or failure of path step */

  int i;

  int nprint = 0, num_total_nodes;

  int numProcUnknowns;
  int *const_delta_s = NULL;
  int step_print;
  double i_print;
  int step_fix = 0; /* What step to fix the problem on */
  int good_mesh = TRUE;
  double *path = NULL, *path1 = NULL;
  double *delta_s = NULL, *delta_s_new = NULL, *delta_s_old = NULL;
  double *delta_s_older = NULL, *delta_s_oldest = NULL;
  double *hDelta_s0 = NULL, *hDelta_s_min = NULL, *hDelta_s_max = NULL;
  double delta_t;
  double theta = 0.0;
  double eps;
  double *lambda = NULL, *lambdaEnd = NULL, *lambdaDelta = NULL;
  double *lambdaLog = NULL, *lambdaRatio = NULL, *lambdaDeltaLog = NULL;
  double hunt_par, dhunt_par, hunt_par_old; /* hunting continuation parameter */
  double dhunt_par_max = 1.0, dhunt_par_min = 0., dhunt_par_0 = 0.1;
  double dhunt_par_new = 0.1, dhunt_par_old;
  int log_ID = -1;
  double timeValueRead = 0.0;

  /*
   * ALC management variables
   */

  int alqALC;
  int *aldALC = NULL;

  /*
   * Other local variables
   */

  int error, err, is_steady_state, inewton;
  int *gindex = NULL, gsize;
  int *p_gsize = NULL;
  double *gvec = NULL;
  double ***gvec_elem;
  FILE *file = NULL;
  double toler_org[3];
  double exo_time = 0;

  struct Results_Description *rd = NULL;

  int tnv;      /* total number of nodal variables and kinds */
  int tev;      /* total number of elem variables and kinds */
  int tnv_post; /* total number of nodal variables and kinds
                           for post processing */
  int tev_post; /* total number of elem variables and kinds
                           for post processing */
  double *gv;

  unsigned int matrix_systems_mask;

  double evol_local = 0.0;
#ifdef PARALLEL
  double evol_global = 0.0;
#endif

  /* Set step_fix only if parallel run and only if fix freq is enabled*/
  if (goma_ctx->num_proc > 1 && goma_ctx->fem.cont->fix_freq > 0) {
    step_fix = 1; /* Always fix on the first timestep to match print frequency */
  }

  static char yo[] = "hunt_problem";

  /*
   * 		BEGIN EXECUTION
   */

  toler_org[0] = goma_ctx->solver.custom_tol1;
  toler_org[1] = goma_ctx->solver.custom_tol2;
  toler_org[2] = goma_ctx->solver.custom_tol3;

  is_steady_state = TRUE;

  p_gsize = &gsize;

  /*
   * set aside space for gather global vectors to print to exoII file
   * note: this is temporary
   *
   * For 2D prototype problem:  allocate space for T, dx, dy arrays
   */

  if (strlen(goma_ctx->soln_out_file)) {
    file = fopen(goma_ctx->soln_out_file, "w");
    if (file == NULL) {
      DPRINTF(stdout, "%s:  opening soln file for writing\n", yo);
      GOMA_EH(GOMA_ERROR, "\t");
    }
  }
#ifdef PARALLEL
  check_parallel_error("Soln output file error");
#endif

  /*
   * Some preliminaries to help setup EXODUS II database output.
   */

  tnv = cnt_nodal_vars();
  /*  tnv_post is calculated in load_nodal_tkn*/
  tev = cnt_elem_vars(exo);
  /*  tev_post is calculated in load_elem_tkn*/

  if (tnv < 0) {
    DPRINTF(stderr, "%s:\tbad tnv.\n", yo);
    GOMA_EH(GOMA_ERROR, "\t");
  }

  if (tev < 0) {
    DPRINTF(stderr, "%s:\tMaybe bad tev? See goma design committee ;) \n", yo);
    /*       exit(-1); */
  }

  rd = (struct Results_Description *)smalloc(sizeof(struct Results_Description));

  if (rd == NULL) {
    GOMA_EH(GOMA_ERROR, "Could not grab Results Description.");
  }
  (void)memset((void *)rd, 0, sizeof(struct Results_Description));

  rd->nev = 0;                          /* number element variables in results */
  rd->ngv = 6 + goma_ctx->problem.n_ac; /* number global variables in results
                        see load_global_var_info for names*/
  rd->nhv = 0;                          /* number history variables in results */

  error = load_global_var_info(rd, 0, "CONV");
  error = load_global_var_info(rd, 1, "NEWT_IT");
  error = load_global_var_info(rd, 2, "MAX_IT");
  error = load_global_var_info(rd, 3, "CONVORDER");
  error = load_global_var_info(rd, 4, "CONVRATE");
  error = load_global_var_info(rd, 5, "MESH_VOLUME");

  if (rd->ngv > MAX_NGV)
    GOMA_EH(-1, "Augmenting condition values overflowing MAX_NGV.  Change and rerun .");

  if (goma_ctx->problem.n_ac > 0) {
    char name[20];

    for (i = 0; i < goma_ctx->problem.n_ac; i++) {
      sprintf(name, "AUGC_%d", i + 1);
      error = load_global_var_info(rd, 6 + i, name);
    }
  }

  gv = alloc_dbl_1(rd->ngv, 0.0);

  /* load nodal types, kinds, names */
  error = load_nodal_tkn(rd, &tnv, &tnv_post); /* load nodal types, kinds, names */

  if (error != 0) {
    DPRINTF(stderr, "%s:  problem with load_nodal_tkn()\n", yo);
    GOMA_EH(GOMA_ERROR, "\t");
  }

  /* load elem types, names */
  error = load_elem_tkn(rd, exo, tev, &tev_post); /* load elem types, names */

  if (error != 0) {
    DPRINTF(stderr, "%s:  problem with load_elem_tkn()\n", yo);
    GOMA_EH(GOMA_ERROR, "\t");
  }

  /*
   * Write out the names of the nodal variables that we will be sending to
   * the EXODUS II output file later.
   */

  gvec_elem = (double ***)calloc(exo->num_elem_blocks, sizeof(double **));
  for (i = 0; i < exo->num_elem_blocks; i++) {
    gvec_elem[i] = (double **)calloc(tev + tev_post, sizeof(double *));
  }

  wr_result_prelim_exo(rd, exo, goma_ctx->exo_file_out, gvec_elem);

  /*
   * This gvec workhorse transports output variables as nodal based vectors
   * that are gather from the solution vector. Note: it is NOT a global
   * vector at all and only carries this processor's nodal variables to
   * the exodus database.
   */

  asdv(&gvec, goma_ctx->mesh.num_node);

  /*
   * Allocate space and manipulate for all the nodes that this processor
   * is aware of...
   */

  num_total_nodes = dpi->num_universe_nodes;

  numProcUnknowns = goma_ctx->problem.num_unknowns[goma_ctx->fem.pg->imtrx] +
                    goma_ctx->problem.num_ext_unknowns[goma_ctx->fem.pg->imtrx];

  /* allocate memory for Volume Constraint Jacobian. ACS 2/99 */

  if (goma_ctx->problem.n_ac > 0) {
    for (iAC = 0; iAC < goma_ctx->problem.n_ac; iAC++) {
      goma_ctx->fem.augc[iAC].d_evol_dx = (double *)malloc(numProcUnknowns * sizeof(double));
    }
  }

  asdv(&resid_vector, numProcUnknowns);
  asdv(&resid_vector_sens, numProcUnknowns);
  asdv(&scale, numProcUnknowns);

  for (i = 0; i < NUM_ALSS; i++) {
    ams[i] = alloc_struct_1(struct GomaLinearSolverData, 1);
  }

#ifdef GOMA_ENABLE_AZTEC
#ifdef MPI
  AZ_set_proc_config(ams[0]->proc_config, MPI_COMM_WORLD);
#else  /* MPI */
  AZ_set_proc_config(ams[0]->proc_config, 0);
#endif /* MPI */
#endif

  /*
   * allocate space for and initialize solution arrays
   */

  asdv(&x, numProcUnknowns);
  asdv(&x_old, numProcUnknowns);
  asdv(&x_older, numProcUnknowns);
  asdv(&x_oldest, numProcUnknowns);
  asdv(&xdot, numProcUnknowns);
  asdv(&xdot_old, numProcUnknowns);
  asdv(&x_update, numProcUnknowns);

  asdv(&x_sens, numProcUnknowns);

  /*
   * Initialize solid inertia flag
   */
  set_solid_inertia();

  /*
   * ALLOCATE ALL THOSE WORK VECTORS FOR HUNTING
   */

  asdv(&lambda, goma_ctx->problem.n_hc);
  asdv(&lambdaEnd, goma_ctx->problem.n_hc);
  asdv(&lambdaDelta, goma_ctx->problem.n_hc);
  asdv(&lambdaLog, goma_ctx->problem.n_hc);
  asdv(&lambdaRatio, goma_ctx->problem.n_hc);
  asdv(&lambdaDeltaLog, goma_ctx->problem.n_hc);
  asdv(&path, goma_ctx->problem.n_hc);
  asdv(&path1, goma_ctx->problem.n_hc);
  asdv(&hDelta_s0, goma_ctx->problem.n_hc);
  asdv(&hDelta_s_min, goma_ctx->problem.n_hc);
  asdv(&hDelta_s_max, goma_ctx->problem.n_hc);
  asdv(&delta_s, goma_ctx->problem.n_hc);
  asdv(&delta_s_new, goma_ctx->problem.n_hc);
  asdv(&delta_s_old, goma_ctx->problem.n_hc);
  asdv(&delta_s_older, goma_ctx->problem.n_hc);
  asdv(&delta_s_oldest, goma_ctx->problem.n_hc);

  aldALC = Ivector_birth(goma_ctx->problem.n_hc);
  const_delta_s = Ivector_birth(goma_ctx->problem.n_hc);

  if (goma_ctx->problem.n_ac > 0) {
    asdv(&x_AC, goma_ctx->problem.n_ac);
    asdv(&x_AC_old, goma_ctx->problem.n_ac);
    asdv(&x_AC_dot, goma_ctx->problem.n_ac);
  }

  /*

   HUNTING BY ZERO AND FIRST ORDER CONTINUATION

  */

  alqALC = 1;

  delta_t = 0.0;
  goma_ctx->fem.tran->delta_t = 0.0; /*for Newmark-Beta terms in Lagrangian Solid*/

  nprint = 0;

  goma_ctx->problem.max_path_steps = goma_ctx->fem.cont->MaxPathSteps;
  eps = goma_ctx->fem.cont->eps;

  for (iHC = 0; iHC < goma_ctx->problem.n_hc; iHC++) {

    const_delta_s[iHC] = 0;

    lambda[iHC] = goma_ctx->fem.hunt[iHC].BegParameterValue;
    lambdaEnd[iHC] = goma_ctx->fem.hunt[iHC].EndParameterValue;
    lambdaDelta[iHC] = lambdaEnd[iHC] - lambda[iHC];
    if (lambda[iHC] > 0. && lambdaEnd[iHC] > 0.0) {
      lambdaLog[iHC] = log10(lambda[iHC]);
      lambdaRatio[iHC] = lambdaEnd[iHC] / lambda[iHC];
      lambdaDeltaLog[iHC] = fabs(log10(lambdaRatio[iHC]));
    } else if (lambda[iHC] < 0. && lambdaEnd[iHC] < 0.0) {
      lambdaLog[iHC] = log10(-lambda[iHC]);
      lambdaRatio[iHC] = lambdaEnd[iHC] / lambda[iHC];
      lambdaDeltaLog[iHC] = fabs(log10(lambdaRatio[iHC]));
    }
    if (goma_ctx->fem.hunt[iHC].ramp == 2) {
      if (log_ID == -1)
        log_ID = iHC;
    }

    if (lambdaDelta[iHC] > 0.0) {
      aldALC[iHC] = +1;
    } else {
      aldALC[iHC] = -1;
    }

    if (goma_ctx->fem.hunt[iHC].ramp == 1) {
      goma_ctx->fem.hunt[iHC].Delta_s0 =
          fabs(lambdaDelta[iHC]) / ((double)(goma_ctx->problem.max_path_steps - 1));
      const_delta_s[iHC] = 1;
    }

    hDelta_s0[iHC] = goma_ctx->fem.hunt[iHC].Delta_s0;
    hDelta_s_min[iHC] = goma_ctx->fem.hunt[iHC].Delta_s_min;
    hDelta_s_max[iHC] = goma_ctx->fem.hunt[iHC].Delta_s_max;

    path[iHC] = path1[iHC] = lambda[iHC];

    if (goma_ctx->debug_flag && goma_ctx->proc_id == 0) {
      fprintf(stdout, "MaxPathSteps: %d \tlambdaEnd: %f\n", goma_ctx->problem.max_path_steps,
              lambdaEnd[iHC]);
      fprintf(stdout, "continuation in progress\n");
    }

    if (hDelta_s0[iHC] > hDelta_s_max[iHC]) {
      hDelta_s0[iHC] = hDelta_s_max[iHC];
    }
    if (hDelta_s0[iHC] < hDelta_s_min[iHC]) {
      hDelta_s0[iHC] = hDelta_s_min[iHC];
    }

    delta_s[iHC] = delta_s_old[iHC] = delta_s_older[iHC] = hDelta_s0[iHC];

    /*
     * ADJUST NATURAL PARAMETER
     */

    update_parameterHC(iHC, path1[iHC], x, xdot, x_AC, delta_s[iHC], cx, exo, dpi);
  }

  /*  define continuation parameter */

  iHC = MAX(0, log_ID);
  dhunt_par = 0.0;
  if (DOUBLE_ZERO(lambdaDelta[iHC])) {
    hunt_par = 1.0;
  } else {
    if (goma_ctx->fem.hunt[iHC].ramp == 2) {
      hunt_par = fabs(log10(fabs(path1[iHC])) - lambdaLog[iHC]) / lambdaDeltaLog[iHC];
      dhunt_par_min = log10(1.0 + hDelta_s_min[iHC] / fabs(lambda[iHC])) / lambdaDeltaLog[iHC];
      dhunt_par_max = log10(1.0 + hDelta_s_max[iHC] / fabs(lambda[iHC])) / lambdaDeltaLog[iHC];
      dhunt_par_0 = log10(1.0 + hDelta_s0[iHC] / fabs(lambda[iHC])) / lambdaDeltaLog[iHC];
    } else {
      hunt_par = (path1[iHC] - lambda[iHC]) / lambdaDelta[iHC];
      dhunt_par_min = aldALC[iHC] * goma_ctx->fem.hunt[iHC].Delta_s_min / lambdaDelta[iHC];
      dhunt_par_max = aldALC[iHC] * goma_ctx->fem.hunt[iHC].Delta_s_max / lambdaDelta[iHC];
      dhunt_par_0 = aldALC[iHC] * goma_ctx->fem.hunt[iHC].Delta_s0 / lambdaDelta[iHC];
    }
    hunt_par = fabs(hunt_par);
  }
  hunt_par_old = hunt_par;
  dhunt_par = dhunt_par_old = dhunt_par_0;
  if (dhunt_par_0 > dhunt_par_max) {
    dhunt_par_0 = dhunt_par_max;
  }

  /*
   *  if compute parameter sensitivities, allocate space for solution
   *  sensitivity vectors
   */

  for (i = 0; i < nn_post_fluxes_sens; i++) {
    num_pvector = MAX(num_pvector, pp_fluxes_sens[i]->vector_id);
  }
  for (i = 0; i < nn_post_data_sens; i++) {
    num_pvector = MAX(num_pvector, pp_data_sens[i]->vector_id);
  }

  if ((nn_post_fluxes_sens + nn_post_data_sens) > 0) {
    num_pvector++;
    num_pvector = MAX(num_pvector, 2);
    x_sens_p = Dmatrix_birth(num_pvector, numProcUnknowns);
  } else {
    x_sens_p = NULL;
  }

  goma_ctx->fem.pg->matrices = malloc(sizeof(struct Matrix_Data));
  goma_ctx->fem.pg->matrices[goma_ctx->fem.pg->imtrx].ams = ams[JAC];
  goma_ctx->fem.pg->matrices[goma_ctx->fem.pg->imtrx].x = x;
  goma_ctx->fem.pg->matrices[goma_ctx->fem.pg->imtrx].x_old = x_old;
  goma_ctx->fem.pg->matrices[goma_ctx->fem.pg->imtrx].x_older = x_older;
  goma_ctx->fem.pg->matrices[goma_ctx->fem.pg->imtrx].xdot = xdot;
  goma_ctx->fem.pg->matrices[goma_ctx->fem.pg->imtrx].xdot_old = xdot_old;
  goma_ctx->fem.pg->matrices[goma_ctx->fem.pg->imtrx].x_update = x_update;
  goma_ctx->fem.pg->matrices[goma_ctx->fem.pg->imtrx].scale = scale;
  goma_ctx->fem.pg->matrices[goma_ctx->fem.pg->imtrx].resid_vector = resid_vector;

  /* Allocate sparse matrix */

  if ((strcmp(goma_ctx->solver.matrix_format, "tpetra") == 0) ||
      (strcmp(goma_ctx->solver.matrix_format, "epetra") == 0)) {
    err = check_compatible_solver();
    GOMA_EH(err, "Incompatible matrix solver for matrix format %s", goma_ctx->solver.matrix_format);
    check_parallel_error("Matrix format / Solver incompatibility");
    GomaSparseMatrix goma_matrix;
    goma_error err =
        GomaSparseMatrix_CreateFromFormat(&goma_matrix, goma_ctx->solver.matrix_format);
    GOMA_EH(err, "GomaSparseMatrix_CreateFromFormat");

    int local_nodes = goma_ctx->mesh.num_internal_nodes + goma_ctx->mesh.num_border_nodes +
                      goma_ctx->mesh.num_external_nodes;
    err = GomaSparseMatrix_SetProblemGraph(
        goma_matrix, goma_ctx->problem.num_internal_dofs[goma_ctx->fem.pg->imtrx],
        goma_ctx->problem.num_boundary_dofs[goma_ctx->fem.pg->imtrx],
        goma_ctx->problem.num_external_dofs[goma_ctx->fem.pg->imtrx], local_nodes, Nodes,
        goma_ctx->problem.max_var_per_node, goma_ctx->problem.matilda, Inter_Mask, exo, dpi, cx,
        goma_ctx->fem.pg->imtrx, goma_ctx->debug_flag, ams[JAC]);
    GOMA_EH(err, "GomaSparseMatrix_SetProblemGraph");

    ams[JAC]->GomaMatrixData = goma_matrix;
    ams[JAC]->npu = goma_ctx->problem.num_internal_dofs[goma_ctx->fem.pg->imtrx] +
                    goma_ctx->problem.num_boundary_dofs[goma_ctx->fem.pg->imtrx];
    ams[JAC]->npu_plus = goma_ctx->problem.num_universe_dofs[goma_ctx->fem.pg->imtrx];

#ifdef GOMA_ENABLE_PETSC
#if PETSC_USE_COMPLEX
  } else if (strcmp(goma_ctx->solver.matrix_format, "petsc_complex") == 0) {
    err = check_compatible_solver();
    GOMA_EH(err, "Incompatible matrix solver for petsc, solver must be petsc");
    check_parallel_error("Matrix format / Solver incompatibility");
    goma_ctx->fem.pg->imtrx = 0;
    goma_error err = goma_setup_petsc_matrix_complex(
        ams[JAC], exo, dpi, x, x_old, xdot, xdot_old,
        goma_ctx->problem.num_internal_dofs[goma_ctx->fem.pg->imtrx],
        goma_ctx->problem.num_boundary_dofs[goma_ctx->fem.pg->imtrx],
        goma_ctx->problem.num_external_dofs[goma_ctx->fem.pg->imtrx], goma_ctx->fem.pg->imtrx);
    GOMA_EH(err, "goma_setup_petsc_matrix");
#else
  } else if (strcmp(goma_ctx->solver.matrix_format, "petsc") == 0) {
    err = check_compatible_solver();
    GOMA_EH(err, "Incompatible matrix solver for petsc, solver must be petsc");
    check_parallel_error("Matrix format / Solver incompatibility");
    goma_ctx->fem.pg->imtrx = 0;
    goma_error err = goma_setup_petsc_matrix(
        ams[JAC], exo, dpi, x, x_old, xdot, xdot_old,
        goma_ctx->problem.num_internal_dofs[goma_ctx->fem.pg->imtrx],
        goma_ctx->problem.num_boundary_dofs[goma_ctx->fem.pg->imtrx],
        goma_ctx->problem.num_external_dofs[goma_ctx->fem.pg->imtrx], goma_ctx->fem.pg->imtrx);
    GOMA_EH(err, "goma_setup_petsc_matrix");
#endif
#endif
  } else if (strcmp(goma_ctx->solver.matrix_format, "msr") == 0) {
    log_msg("alloc_MSR_sparse_arrays...");
    alloc_MSR_sparse_arrays(&ija, &a, &a_old, 0, node_to_fill, exo, dpi);
    /*
     * An attic to store external dofs column names is needed when
     * running in parallel.
     */
    alloc_extern_ija_buffer(goma_ctx->problem.num_universe_dofs[goma_ctx->fem.pg->imtrx],
                            goma_ctx->problem.num_internal_dofs[goma_ctx->fem.pg->imtrx] +
                                goma_ctx->problem.num_boundary_dofs[goma_ctx->fem.pg->imtrx],
                            ija, &ija_attic);

    /*
     * Any necessary one time initialization of the linear
     * solver package (Aztec).
     */

    ams[JAC]->bindx = ija;
    ams[JAC]->val = a;
    ams[JAC]->belfry = ija_attic;
    ams[JAC]->val_old = a_old;

    /*
     * These point to nowhere since we're using MSR instead of VBR
     * format.
     */

    ams[JAC]->indx = NULL;
    ams[JAC]->bpntr = NULL;
    ams[JAC]->rpntr = NULL;
    ams[JAC]->cpntr = NULL;
    ams[JAC]->npn = dpi->num_internal_nodes + dpi->num_boundary_nodes;
    ams[JAC]->npn_plus =
        dpi->num_internal_nodes + dpi->num_boundary_nodes + dpi->num_external_nodes;

    ams[JAC]->npu = goma_ctx->problem.num_internal_dofs[goma_ctx->fem.pg->imtrx] +
                    goma_ctx->problem.num_boundary_dofs[goma_ctx->fem.pg->imtrx];
    ams[JAC]->npu_plus = goma_ctx->problem.num_universe_dofs[goma_ctx->fem.pg->imtrx];

    ams[JAC]->nnz = ija[goma_ctx->problem.num_internal_dofs[goma_ctx->fem.pg->imtrx] +
                        goma_ctx->problem.num_boundary_dofs[goma_ctx->fem.pg->imtrx]] -
                    1;
    ams[JAC]->nnz_plus = ija[goma_ctx->problem.num_universe_dofs[goma_ctx->fem.pg->imtrx]];

  } else if (strcmp(goma_ctx->solver.matrix_format, "vbr") == 0) {
    log_msg("alloc_VBR_sparse_arrays...");
    alloc_VBR_sparse_arrays(ams[JAC], exo, dpi);
    ija_attic = NULL;
    ams[JAC]->belfry = ija_attic;

    a = ams[JAC]->val;
    if (!goma_ctx->solver.save_old_a)
      a_old = ams[JAC]->val_old;
  } else if (strcmp(goma_ctx->solver.matrix_format, "front") == 0) {
    /* Don't allocate any sparse matrix space when using front */
    ams[JAC]->bindx = NULL;
    ams[JAC]->val = NULL;
    ams[JAC]->belfry = NULL;
    ams[JAC]->val_old = NULL;
    ams[JAC]->indx = NULL;
    ams[JAC]->bpntr = NULL;
    ams[JAC]->rpntr = NULL;
    ams[JAC]->cpntr = NULL;

  } else {
    GOMA_EH(GOMA_ERROR, "Attempted to allocate unknown sparse matrix format");
  }

  init_vec(x, cx, exo, dpi, x_AC, goma_ctx->problem.n_ac, &timeValueRead);

  /*  if read ACs, update data floats */
  if (goma_ctx->problem.n_ac > 0) {
    if (goma_ctx->fem.augc[0].iread == 1) {
      for (iAC = 0; iAC < goma_ctx->problem.n_ac; iAC++) {
        update_parameterAC(iAC, x, xdot, x_AC, cx, exo, dpi);
      }
    }
  }

  /*
   * set boundary conditions on the initial conditions
   */

  find_and_set_Dirichlet(x, xdot, exo, dpi);

  exchange_dof(cx, dpi, x, goma_ctx->fem.pg->imtrx);

  dcopy1(numProcUnknowns, x, x_old);
  dcopy1(numProcUnknowns, x_old, x_older);
  dcopy1(numProcUnknowns, x_older, x_oldest);

  if (goma_ctx->problem.n_ac > 0) {
    dcopy1(goma_ctx->problem.n_ac, x_AC, x_AC_old);
  }

  /*
   * initialize the counters for when to print out data
   */

  step_print = 1;

  matrix_systems_mask = 1;

  log_msg("sl_init()...");
  sl_init(matrix_systems_mask, ams, exo, dpi, cx);

#ifdef PARALLEL
  /*
   * Make sure the solver was properly initialized on all processors.
   */
  check_parallel_error("Solver initialization problems");
#endif

#ifdef GOMA_ENABLE_AZTEC
  ams[JAC]->options[AZ_keep_info] = 1;
#endif

  DPRINTF(stdout, "\nINITIAL ELEMENT QUALITY CHECK---\n");
  good_mesh = element_quality(exo, x, ams[0]->proc_config);

  if (goma_ctx->solver.output_variable_stats) {
    err = variable_stats(x, path1[0], goma_ctx->solver.output_variable_regression);
    GOMA_EH(err, "Problem with variable_stats!");
    if (goma_ctx->proc_id == 0)
      fflush(stdout);
  }

  /*
   * set the number of successful path steps to zero
   */

  nt = 0;

  /*
   * LOOP THROUGH PARAMETER UNTIL MAX NUMBER
   * OF STEPS SURPASSED
   */

  if (goma_ctx->problem.n_ac > 0) {
    dcopy1(goma_ctx->problem.n_ac, x_AC, &(gv[5]));
  }

  for (n = 0; n < goma_ctx->problem.max_path_steps; n++) {

    alqALC = 1;

    for (iHC = 0; iHC < goma_ctx->problem.n_hc; iHC++) {

      switch (aldALC[iHC]) {
      case -1: /* REDUCING PARAMETER DIRECTION */
        if (path1[iHC] <= lambdaEnd[iHC]) {
          alqALC = -1;
          path1[iHC] = lambdaEnd[iHC];
          delta_s[iHC] = path[iHC] - path1[iHC];
        }
        break;
      case +1: /* RISING PARAMETER DIRECTION */
        if (path1[iHC] >= lambdaEnd[iHC]) {
          alqALC = -1;
          path1[iHC] = lambdaEnd[iHC];
          delta_s[iHC] = path1[iHC] - path[iHC];
        }
        break;
      }
    } /*  end of iHC loop */

    /*
     * ADJUST NATURAL PARAMETER
     */

    for (iHC = 0; iHC < goma_ctx->problem.n_hc; iHC++) {
      update_parameterHC(iHC, path1[iHC], x, xdot, x_AC, delta_s[iHC], cx, exo, dpi);
    } /*  end of iHC loop */

    iHC = MAX(0, log_ID);
    if (DOUBLE_ZERO(lambdaDelta[iHC])) {
      hunt_par = 1.0;
    } else {
      if (goma_ctx->fem.hunt[iHC].ramp == 2) {
        hunt_par = fabs(log10(fabs(path1[iHC])) - lambdaLog[iHC]) / lambdaDeltaLog[iHC];
        if (n > 0) {
          dhunt_par = fabs(log10(fabs(path1[iHC])) - log10(fabs(path[iHC]))) / lambdaDeltaLog[iHC];
          dhunt_par_min = log10(1.0 + hDelta_s_min[iHC] / fabs(path1[iHC])) / lambdaDeltaLog[iHC];
          dhunt_par_max = log10(1.0 + hDelta_s_max[iHC] / fabs(path1[iHC])) / lambdaDeltaLog[iHC];
        }
      } else {
        hunt_par = (path1[iHC] - lambda[iHC]) / lambdaDelta[iHC];
        if (n > 0) {
          dhunt_par = (path1[iHC] - path[iHC]) / lambdaDelta[iHC];
        }
      }
    }
    for (iHC = 0; iHC < goma_ctx->problem.n_hc; iHC++) {
      if (goma_ctx->fem.hunt[iHC].ramp == 2) {
        delta_s[iHC] = -fabs(lambda[iHC]) * pow(lambdaRatio[iHC], hunt_par - dhunt_par);
        delta_s[iHC] += fabs(lambda[iHC]) * pow(lambdaRatio[iHC], hunt_par);
      } else {
        delta_s[iHC] = dhunt_par * lambdaDelta[iHC];
      }
    }

    /*
     * IF STEP CHANGED, REDO FIRST ORDER PREDICTION
     */
    if (hunt_par >= 1.0) {
      alqALC = -1;
    }

    if (alqALC == -1) {
      DPRINTF(stdout, "\n\t ******** LAST PATH STEP!\n");
      dcopy1(numProcUnknowns, x_old, x);

      switch (goma_ctx->problem.continuation) {
      case HUN_ZEROTH:
        break;
      case HUN_FIRST:
        v1add(numProcUnknowns, &x[0], dhunt_par, &x_sens[0]);
        break;
      }
    }

    /*
     * reset Dirichlet condition Mask, node->DBC to -1 where it
     * is set in order for Dirichlet conditions to be
     * set appropriately for each path step
     */

    nullify_dirichlet_bcs();

    find_and_set_Dirichlet(x, xdot, exo, dpi);

    exchange_dof(cx, dpi, x, goma_ctx->fem.pg->imtrx);

    if (goma_ctx->proc_id == 0) {
      DPRINTF(stdout, "\n\t----------------------------------");
      switch (goma_ctx->problem.continuation) {
      case HUN_ZEROTH:
        DPRINTF(stdout, "\n\tZero Order Hunting:");
        break;
      case HUN_FIRST:
        DPRINTF(stdout, "\n\tFirst Order Hunting:");
        break;
      }
      DPRINTF(stdout, "\n\tStep number: %4d of %4d (max)", n + 1, goma_ctx->problem.max_path_steps);
      DPRINTF(stdout, "\n\tAttempting solution at: theta = %g ;  step = %g", hunt_par, dhunt_par);
      for (iHC = 0; iHC < goma_ctx->problem.n_hc; iHC++) {
        switch (goma_ctx->fem.hunt[iHC].Type) {
        case 1: /* BC */
          DPRINTF(stdout, "\n\tBCID=%3d DFID=%5d", goma_ctx->fem.hunt[iHC].BCID,
                  goma_ctx->fem.hunt[iHC].DFID);
          break;
        case 2: /* MT */
          DPRINTF(stdout, "\n\tMTID=%3d MPID=%5d", goma_ctx->fem.hunt[iHC].MTID + 1,
                  goma_ctx->fem.hunt[iHC].MPID);
          break;
        case 3: /* AC */
          DPRINTF(stdout, "\n\tACID=%3d DFID=%5d", goma_ctx->fem.hunt[iHC].BCID,
                  goma_ctx->fem.hunt[iHC].DFID);
          break;
        }
        DPRINTF(stdout, " Parameter= % 10.6e delta_s= %10.6e", path1[iHC], delta_s[iHC]);
      }
    }

    ni = 0;
    do {

      err = solve_nonlinear_problem(ams[JAC], x, delta_t, theta, x_old, x_older, xdot, xdot_old,
                                    resid_vector, x_update, scale, &converged, &nprint, tev,
                                    tev_post, gv, rd, gindex, p_gsize, gvec, gvec_elem, path1[0],
                                    exo, dpi, cx, 0, &path_step_reform, is_steady_state, x_AC,
                                    x_AC_dot, hunt_par, resid_vector_sens, x_sens, x_sens_p, NULL);

      if (err == -1)
        converged = 0;
      inewton = err;
      if (converged) {
        GOMA_EH(error, "error writing ASCII soln file."); /* srs need to check */

        if (goma_ctx->write_intermediate_solutions == 0) {
          exo_time = aldALC[0] * path1[0];
          write_solution(goma_ctx->exo_file_out, resid_vector, x, x_sens_p, x_old, xdot, xdot_old,
                         tev, tev_post, gv, rd, gvec, gvec_elem, &nprint, delta_s[0], theta,
                         exo_time, NULL, exo, dpi);
        }

        /*
         * PRINT OUT VALUES OF EXTRA UNKNOWNS
         * FROM AUGMENTING CONDITIONS
         */

        if (goma_ctx->problem.n_ac > 0) {

          DPRINTF(stdout, "\n------------------------------\n");
          DPRINTF(stdout, "Augmenting Conditions:    %4d\n", goma_ctx->problem.n_ac);
          DPRINTF(stdout, "Number of extra unknowns: %4d\n\n", goma_ctx->problem.n_ac);

          for (iAC = 0; iAC < goma_ctx->problem.n_ac; iAC++) {
            evol_local = goma_ctx->fem.augc[iAC].evol;
#ifdef PARALLEL
            if (goma_ctx->num_proc > 1 && (goma_ctx->fem.augc[iAC].Type == AC_VOLUME ||
                                           goma_ctx->fem.augc[iAC].Type == AC_POSITION ||
                                           goma_ctx->fem.augc[iAC].Type == AC_ANGLE ||
                                           goma_ctx->fem.augc[iAC].Type == AC_POSITION_MT)) {
              MPI_Allreduce(&evol_local, &evol_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
              evol_local = evol_global;
            }
#endif
            if (goma_ctx->fem.augc[iAC].Type == AC_USERBC) {
              DPRINTF(stdout, "\tAC[%4d] DF[%4d] = %10.6e\n", goma_ctx->fem.augc[iAC].BCID,
                      goma_ctx->fem.augc[iAC].DFID, x_AC[iAC]);
            } else if (goma_ctx->fem.augc[iAC].Type == AC_USERMAT ||
                       goma_ctx->fem.augc[iAC].Type == AC_FLUX_MAT) {
              DPRINTF(stdout, "\n MT[%4d] MP[%4d] = %10.6e\n", goma_ctx->fem.augc[iAC].MTID,
                      goma_ctx->fem.augc[iAC].MPID, x_AC[iAC]);
            } else if (goma_ctx->fem.augc[iAC].Type == AC_VOLUME) {
              DPRINTF(stdout, "\tMT[%4d] VC[%4d]=%10.6e Param=%10.6e\n",
                      goma_ctx->fem.augc[iAC].MTID, goma_ctx->fem.augc[iAC].VOLID, evol_local,
                      x_AC[iAC]);
            } else if (goma_ctx->fem.augc[iAC].Type == AC_POSITION ||
                       goma_ctx->fem.augc[iAC].Type == AC_POSITION_MT) {
              DPRINTF(stdout, "\tMT[%4d] XY[%4d]=%10.6e Param=%10.6e\n",
                      goma_ctx->fem.augc[iAC].MTID, goma_ctx->fem.augc[iAC].VOLID, evol_local,
                      x_AC[iAC]);
            } else if (goma_ctx->fem.augc[iAC].Type == AC_ANGLE) {
              evol_local = goma_ctx->fem.augc[iAC].lm_resid + goma_ctx->fem.augc[iAC].CONSTV;
              DPRINTF(stdout, "\tNodeSet[%4d]_Ang = %g F_bal = %6.3e VC[%4d] Param=%6.3e\n",
                      goma_ctx->fem.augc[iAC].MTID, evol_local, goma_ctx->fem.augc[iAC].lm_resid,
                      goma_ctx->fem.augc[iAC].VOLID, x_AC[iAC]);
            } else if (goma_ctx->fem.augc[iAC].Type == AC_FLUX) {
              DPRINTF(stdout, "\tBC[%4d] DF[%4d]=%10.6e\n", goma_ctx->fem.augc[iAC].BCID,
                      goma_ctx->fem.augc[iAC].DFID, x_AC[iAC]);
            }
          }
        }

        /* Check element quality */
        good_mesh = element_quality(exo, x, ams[0]->proc_config);

        /*

          INTEGRATE FLUXES, FORCES

        */

        for (i = 0; i < nn_post_fluxes; i++) {
          evaluate_flux(exo, dpi, pp_fluxes[i]->ss_id, pp_fluxes[i]->flux_type,
                        pp_fluxes[i]->flux_type_name, pp_fluxes[i]->blk_id,
                        pp_fluxes[i]->species_number, pp_fluxes[i]->flux_filenm,
                        pp_fluxes[i]->profile_flag, x, xdot, NULL, delta_s[0], path1[0], 1);
        }

        /*
          COMPUTE FLUX, FORCE SENSITIVITIES
        */

        for (i = 0; i < nn_post_fluxes_sens; i++) {
          evaluate_flux_sens(exo, dpi, pp_fluxes_sens[i]->ss_id, pp_fluxes_sens[i]->flux_type,
                             pp_fluxes_sens[i]->flux_type_name, pp_fluxes_sens[i]->blk_id,
                             pp_fluxes_sens[i]->species_number, pp_fluxes_sens[i]->sens_type,
                             pp_fluxes_sens[i]->sens_id, pp_fluxes_sens[i]->sens_flt,
                             pp_fluxes_sens[i]->sens_flt2, pp_fluxes_sens[i]->vector_id,
                             pp_fluxes_sens[i]->flux_filenm, pp_fluxes_sens[i]->profile_flag, x,
                             xdot, x_sens_p, delta_s[0], path1[0], 1);
        }
        /*
         * Compute global volumetric quantities
         */
        for (i = 0; i < nn_volume; i++) {
          evaluate_volume_integral(
              exo, dpi, pp_volume[i]->volume_type, pp_volume[i]->volume_name, pp_volume[i]->blk_id,
              pp_volume[i]->species_no, pp_volume[i]->volume_fname, pp_volume[i]->params,
              pp_volume[i]->num_params, NULL, x, xdot, delta_s[0], path1[0], 1);
        }

      } /* end of if converged block */

      /*
       * INCREMENT COUNTER
       */

      ni++;

      /*
       *
       * DID IT CONVERGE ?
       * IF NOT, REDUCE STEP SIZE AND TRY AGAIN
       *
       */

      if (!converged) {

        if (ni > goma_ctx->solver.max_newton_steps / 2) {
          DPRINTF(stdout, "\n ************************************\n");
          DPRINTF(stdout, " W: Did not converge in Newton steps.\n");
          DPRINTF(stdout, "    Find better initial guess.       \n");
          DPRINTF(stdout, " ************************************\n");
          goto free_and_clear;
        }

        /*
         * ADJUST STEP SIZE - unless failed on first step
         */

        if (nt != 0) {
          DPRINTF(stdout, "\n\tFailed to converge:\n");

          dhunt_par *= 0.5;
          hunt_par = hunt_par_old + dhunt_par;
          for (iHC = 0; iHC < goma_ctx->problem.n_hc; iHC++) {
            if (goma_ctx->fem.hunt[iHC].ramp == 2) {
              path1[iHC] = lambda[iHC] * pow(lambdaRatio[iHC], hunt_par);
            } else if (goma_ctx->fem.hunt[iHC].ramp == 1) {
              delta_s[iHC] *= 0.5;
              path1[iHC] = path[iHC] - ((double)aldALC[iHC]) * delta_s[iHC];
            } else {
              path1[iHC] = lambda[iHC] + hunt_par * lambdaDelta[iHC];
            }
          }

          alqALC = 1;

          DPRINTF(stdout, "Decreasing step-length to %10.6e.\n", dhunt_par);

          if (dhunt_par < dhunt_par_min) {
            DPRINTF(stdout, "\n X: C step-length reduced below minimum.");
            DPRINTF(stdout, "\n theta: %g ; delta theta %g ; delta theta_min: %g", hunt_par,
                    dhunt_par, dhunt_par_min);
            DPRINTF(stdout, "\n    Program terminated.\n");
            /* This needs to have a return value of 0, indicating
             * success, for the continuation script to not treat this
             * as a failed command. */
            goto free_and_clear;
          }
#ifdef PARALLEL
          check_parallel_error("\t");
#endif

          /*
           * ADJUST NATURAL PARAMETER
           */

          for (iHC = 0; iHC < goma_ctx->problem.n_hc; iHC++) {
            update_parameterHC(iHC, path1[iHC], x, xdot, x_AC, delta_s[iHC], cx, exo, dpi);
          } /* end of iHC loop  */

          iHC = MAX(0, log_ID);
          if (DOUBLE_ZERO(lambdaDelta[iHC])) {
            hunt_par = 1.0;
          } else {
            if (goma_ctx->fem.hunt[iHC].ramp == 2) {
              hunt_par = fabs(log10(fabs(path1[iHC])) - lambdaLog[iHC]) / lambdaDeltaLog[iHC];
            } else {
              hunt_par = (path1[iHC] - lambda[iHC]) / lambdaDelta[iHC];
            }
            dhunt_par = hunt_par - hunt_par_old;
          }

          /*
           * GET ZERO OR FIRST ORDER PREDICTION
           */

          switch (goma_ctx->problem.continuation) {
          case HUN_ZEROTH:
            vcopy(numProcUnknowns, &x[0], 1.0, &x_old[0]);
            break;
          case HUN_FIRST:
            v2sum(numProcUnknowns, &x[0], 1.0, &x_old[0], dhunt_par, &x_sens[0]);
            break;
          }

          /* MMH: Needed to put this in, o/w it may find that the
           * solution and residual HAPPEN to satisfy the convergence
           * criterion for the next newton solve...
           */
          find_and_set_Dirichlet(x, xdot, exo, dpi);

          exchange_dof(cx, dpi, x, goma_ctx->fem.pg->imtrx);

          if (goma_ctx->problem.n_ac > 0) {
            dcopy1(goma_ctx->problem.n_ac, x_AC_old, x_AC);
            for (iAC = 0; iAC < goma_ctx->problem.n_ac; iAC++) {
              update_parameterAC(iAC, x, xdot, x_AC, cx, exo, dpi);
            }
          }

          iHC = MAX(0, log_ID);
          if (DOUBLE_ZERO(lambdaDelta[iHC])) {
            hunt_par = 1.0;
          } else {
            if (goma_ctx->fem.hunt[iHC].ramp == 2) {
              hunt_par = fabs(log10(fabs(path1[iHC])) - lambdaLog[iHC]) / lambdaDeltaLog[iHC];
            } else {
              hunt_par = (path1[iHC] - lambda[iHC]) / lambdaDelta[iHC];
            }
          }
        } else if (inewton == -1) {
          DPRINTF(stdout, "\nHmm... trouble on first step \n  Let's try some more relaxation  \n");
          if ((goma_ctx->solver.damp_factor1 <= 1. && goma_ctx->solver.damp_factor1 >= 0.) &&
              (goma_ctx->solver.damp_factor2 <= 1. && goma_ctx->solver.damp_factor2 >= 0.) &&
              (goma_ctx->solver.damp_factor3 <= 1. && goma_ctx->solver.damp_factor3 >= 0.)) {
            goma_ctx->solver.custom_tol1 *= 0.01;
            goma_ctx->solver.custom_tol2 *= 0.01;
            goma_ctx->solver.custom_tol3 *= 0.01;
            DPRINTF(stdout, "  custom tolerances %g %g %g  \n", goma_ctx->solver.custom_tol1,
                    goma_ctx->solver.custom_tol2, goma_ctx->solver.custom_tol3);
          } else {
            goma_ctx->solver.damp_factor1 *= 0.5;
            DPRINTF(stdout, "  damping factor %g  \n", goma_ctx->solver.damp_factor1);
          }

          vcopy(numProcUnknowns, &x[0], 1.0, &x_old[0]);

          /* MMH: Needed to put this in, o/w it may find that the
           * solution and residual HAPPEN to satisfy the convergence
           * criterion for the next newton solve...
           */
          find_and_set_Dirichlet(x, xdot, exo, dpi);

          exchange_dof(cx, dpi, x, goma_ctx->fem.pg->imtrx);

          if (goma_ctx->problem.n_ac > 0) {
            dcopy1(goma_ctx->problem.n_ac, x_AC_old, x_AC);
            for (iAC = 0; iAC < goma_ctx->problem.n_ac; iAC++) {
              update_parameterAC(iAC, x, xdot, x_AC, cx, exo, dpi);
            }
          }

        } else {
          DPRINTF(stdout,
                  "\nHmm... could not converge on first step\n Let's try some more iterations\n");
          if ((goma_ctx->solver.damp_factor1 <= 1. && goma_ctx->solver.damp_factor1 >= 0.) &&
              (goma_ctx->solver.damp_factor2 <= 1. && goma_ctx->solver.damp_factor2 >= 0.) &&
              (goma_ctx->solver.damp_factor3 <= 1. && goma_ctx->solver.damp_factor3 >= 0.)) {
            if (goma_ctx->fem.hunt[0].BCID == -1) {
              if (goma_ctx->write_intermediate_solutions == 0) {
                write_solution(goma_ctx->exo_file_out, resid_vector, x, x_sens_p, x_old, xdot,
                               xdot_old, tev, tev_post, gv, rd, gvec, gvec_elem, &nprint,
                               delta_s[0], theta, goma_ctx->solver.custom_tol1, NULL, exo, dpi);
                nprint++;
              }
            }
            goma_ctx->solver.custom_tol1 *= 100.;
            goma_ctx->solver.custom_tol2 *= 100.;
            goma_ctx->solver.custom_tol3 *= 100.;
            DPRINTF(stdout, "  custom tolerances %g %g %g  \n", goma_ctx->solver.custom_tol1,
                    goma_ctx->solver.custom_tol2, goma_ctx->solver.custom_tol3);
          } else {
            if (goma_ctx->fem.hunt[0].BCID == -1) {
              if (goma_ctx->write_intermediate_solutions == 0) {
                DPRINTF(stdout, "  writing solution %g  \n", goma_ctx->solver.damp_factor1);
                write_solution(goma_ctx->exo_file_out, resid_vector, x, x_sens_p, x_old, xdot,
                               xdot_old, tev, tev_post, gv, rd, gvec, gvec_elem, &nprint,
                               delta_s[0], theta, goma_ctx->solver.damp_factor1, NULL, exo, dpi);
                nprint++;
              }
            }
            goma_ctx->solver.damp_factor1 *= 2.0;
            goma_ctx->solver.damp_factor1 = MIN(goma_ctx->solver.damp_factor1, 1.0);
            DPRINTF(stdout, "  damping factor %g  \n", goma_ctx->solver.damp_factor1);
          }
        }

      } /* end of !converged */

    } while (converged == 0);

    /*
     * CONVERGED
     */
    nt++;
    goma_ctx->solver.custom_tol1 = toler_org[0];
    goma_ctx->solver.custom_tol2 = toler_org[1];
    goma_ctx->solver.custom_tol3 = toler_org[2];
    goma_ctx->solver.damp_factor1 = 1.0;
    DPRINTF(stdout, "\n\tStep accepted, theta (proportion complete) = %10.6e\n", hunt_par);
    for (iHC = 0; iHC < goma_ctx->problem.n_hc; iHC++) {
      switch (goma_ctx->fem.hunt[iHC].Type) {
      case 1: /* BC */
        DPRINTF(stdout, "\tStep accepted, BCID=%3d DFID=%5d", goma_ctx->fem.hunt[iHC].BCID,
                goma_ctx->fem.hunt[iHC].DFID);
        break;
      case 2: /* MT */
        DPRINTF(stdout, "\tStep accepted, MTID=%3d MPID=%5d", goma_ctx->fem.hunt[iHC].MTID + 1,
                goma_ctx->fem.hunt[iHC].MPID);
        break;
      case 3: /* AC */
        DPRINTF(stdout, "\tStep accepted, ACID=%3d DFID=%5d", goma_ctx->fem.hunt[iHC].BCID,
                goma_ctx->fem.hunt[iHC].DFID);
        break;
      }
      DPRINTF(stdout, " Parameter= % 10.6e\n", path1[iHC]);
    }

    /*
     * check path step error, if too large do not enlarge path step
     */

    iHC = MAX(0, log_ID);
    if ((ni == 1) && (n != 0) && (!const_delta_s[iHC])) {
      dhunt_par_new = path_step_control(num_total_nodes, dhunt_par, dhunt_par_old, x, eps,
                                        &success_ds, goma_ctx->fem.cont->use_var_norm, inewton);
      if (dhunt_par_new > dhunt_par_max) {
        dhunt_par_new = dhunt_par_max;
      }
    } else {
      success_ds = 1;
      dhunt_par_new = dhunt_par;
    }

    /*
     * determine whether to print out the data or not
     */

    i_print = 0;
    if (nt == step_print) {
      i_print = 1;
      step_print += goma_ctx->fem.cont->print_freq;
    }

    if (alqALC == -1) {
      i_print = 1;
    }

    if (i_print) {
      error = write_ascii_soln(x, resid_vector, numProcUnknowns, x_AC, goma_ctx->problem.n_ac,
                               path1[0], file);
      if (error) {
        DPRINTF(stderr, "%s:  error writing ASCII soln file\n", yo);
      }
      if (goma_ctx->write_intermediate_solutions == 0) {
        exo_time = aldALC[0] * path1[0];
        write_solution(goma_ctx->exo_file_out, resid_vector, x, x_sens_p, x_old, xdot, xdot_old,
                       tev, tev_post, gv, rd, gvec, gvec_elem, &nprint, delta_s[0], theta, exo_time,
                       NULL, exo, dpi);
        nprint++;
        if (goma_ctx->anneal_mesh && goma_ctx->fem.cont->anneal_on_print) {
          err = anneal_mesh(x, tev, tev_post, NULL, rd, path1[0], exo, dpi);
          GOMA_EH(err, "anneal_mesh() bad return.");
        }
      }
    }

    if (goma_ctx->solver.output_variable_stats) {
      err = variable_stats(x, path1[0], goma_ctx->solver.output_variable_regression);
      GOMA_EH(err, "Problem with variable_stats!");
      if (goma_ctx->proc_id == 0)
        fflush(stdout);
    }

#ifdef PARALLEL
    if (step_fix != 0 && nt == step_fix) {
      /* Barrier because fix needs both files to be finished printing
         and fix always occurs on the same timestep as printing */
      MPI_Barrier(MPI_COMM_WORLD);
      fix_output();
      /* Fix step is relative to print step */
      step_fix += goma_ctx->fem.cont->fix_freq * goma_ctx->fem.cont->print_freq;
    }
#endif

    /*
     * backup old solutions
     * can use previous solutions for prediction one day
     */

    dcopy1(numProcUnknowns, x_older, x_oldest);
    dcopy1(numProcUnknowns, x_old, x_older);
    dcopy1(numProcUnknowns, x, x_old);

    dcopy1(goma_ctx->problem.n_hc, delta_s_older, delta_s_oldest);
    dcopy1(goma_ctx->problem.n_hc, delta_s_old, delta_s_older);
    dcopy1(goma_ctx->problem.n_hc, delta_s, delta_s_old);
    dcopy1(goma_ctx->problem.n_hc, delta_s_new, delta_s);
    dhunt_par_old = dhunt_par;
    dhunt_par = dhunt_par_new;
    hunt_par_old = hunt_par;
    if (goma_ctx->problem.n_ac > 0) {
      dcopy1(goma_ctx->problem.n_ac, x_AC, x_AC_old);
    }

    /*
     * INCREMENT/DECREMENT PARAMETER
     */

    hunt_par += dhunt_par;
    if (hunt_par > 1.0) {
      dhunt_par = 1.0 - (hunt_par - dhunt_par);
      hunt_par = 1.0;
    }
    for (iHC = 0; iHC < goma_ctx->problem.n_hc; iHC++) {
      if (goma_ctx->fem.hunt[iHC].ramp == 2) {
        path1[iHC] = lambda[iHC] * pow(lambdaRatio[iHC], hunt_par);
        path[iHC] = lambda[iHC] * pow(lambdaRatio[iHC], hunt_par - dhunt_par);
      } else {
        path1[iHC] = lambda[iHC] + hunt_par * lambdaDelta[iHC];
        path[iHC] = lambda[iHC] + (hunt_par - dhunt_par) * lambdaDelta[iHC];
      }
    }
    /*
     * ADJUST NATURAL PARAMETER
     */

    for (iHC = 0; iHC < goma_ctx->problem.n_hc; iHC++) {
      update_parameterHC(iHC, path1[iHC], x, xdot, x_AC, delta_s[iHC], cx, exo, dpi);
    } /*  end of iHC loop */

    /*
     * GET FIRST ORDER PREDICTION
     */

    switch (goma_ctx->problem.continuation) {
    case HUN_ZEROTH:
      break;
    case HUN_FIRST:
      v1add(numProcUnknowns, &x[0], dhunt_par, &x_sens[0]);
      break;
    }

    if (!good_mesh)
      goto free_and_clear;

    /*
     *
     * CHECK END CONTINUATION
     *
     */

    if (alqALC == -1) {
      alqALC = 0;
    } else {
      alqALC = 1;
    }

    if (alqALC == 0) {
      DPRINTF(stdout, "\n\n\t I will continue no more!\n\t No more continuation for you!\n");
      goto free_and_clear;
    }

  } /* n */

  if (n == goma_ctx->problem.max_path_steps && aldALC[0] * (lambdaEnd[0] - path[0]) > 0) {
    DPRINTF(
        stderr,
        "\n\tFailed to reach end of hunt in maximum number of successful steps (%d).\n\tSorry.\n",
        goma_ctx->problem.max_path_steps);
    goto free_and_clear;
    /*exit(0);  */
  }
#ifdef PARALLEL
  check_parallel_error("Hunting error");
#endif

  /*
   * DONE CONTINUATION
   */

free_and_clear:

  /*
   * Transform the node point coordinates according to the
   * displacements and write out all the results using the
   * displaced coordinates. Set the displacement field to
   * zero, too.
   */

  if (goma_ctx->anneal_mesh) {
    err = anneal_mesh(x, tev, tev_post, NULL, rd, path1[0], exo, dpi);
    GOMA_EH(err, "anneal_mesh() bad return.");
  }

  /*
   * Free a bunch of variables that aren't needed anymore
   */
  safer_free((void **)&goma_ctx->bc.rot_types);
  safer_free((void **)&node_to_fill);

  safer_free((void **)&resid_vector);
  safer_free((void **)&resid_vector_sens);
  safer_free((void **)&scale);
  safer_free((void **)&x);

  if (goma_ctx->problem.n_ac > 0) {
    safer_free((void **)&x_AC);
    safer_free((void **)&x_AC_old);
    safer_free((void **)&x_AC_dot);
  }

  safer_free((void **)&x_old);
  safer_free((void **)&x_older);
  safer_free((void **)&x_oldest);
  safer_free((void **)&xdot);
  safer_free((void **)&xdot_old);
  safer_free((void **)&x_update);

  safer_free((void **)&x_sens);

  if ((nn_post_data_sens + nn_post_fluxes_sens) > 0)
    Dmatrix_death(x_sens_p, num_pvector, numProcUnknowns);

  for (i = 0; i < MAX_NUMBER_MATLS; i++) {
    for (n = 0; n < MAX_MODES; n++) {
      safer_free((void **)&(goma_ctx->fem.ve_glob[i][n]->gn));
      safer_free((void **)&(goma_ctx->fem.ve_glob[i][n]->time_const_st));
      safer_free((void **)&(goma_ctx->fem.ve_glob[i][n]));
    }
    safer_free((void **)&(goma_ctx->fem.vn_glob[i]));
  }

  sl_free(matrix_systems_mask, ams);

  for (i = 0; i < NUM_ALSS; i++) {
    safer_free((void **)&(ams[i]));
  }

  safer_free((void **)&gvec);

  safer_free((void **)&lambda);
  safer_free((void **)&lambdaEnd);
  safer_free((void **)&lambdaDelta);
  safer_free((void **)&lambdaLog);
  safer_free((void **)&lambdaRatio);
  safer_free((void **)&lambdaDeltaLog);
  safer_free((void **)&path);
  safer_free((void **)&path1);
  safer_free((void **)&hDelta_s0);
  safer_free((void **)&hDelta_s_min);
  safer_free((void **)&hDelta_s_max);
  safer_free((void **)&delta_s);
  safer_free((void **)&delta_s_new);
  safer_free((void **)&delta_s_old);
  safer_free((void **)&delta_s_older);
  safer_free((void **)&delta_s_oldest);

  Ivector_death(&aldALC[0], goma_ctx->problem.n_hc);
  Ivector_death(&const_delta_s[0], goma_ctx->problem.n_hc);

  i = 0;
  for (eb_indx = 0; eb_indx < exo->num_elem_blocks; eb_indx++) {
    for (ev_indx = 0; ev_indx < rd->nev; ev_indx++) {
      if (exo->elem_var_tab[i++] == 1) {
        safer_free((void **)&(gvec_elem[eb_indx][ev_indx]));
      }
    }
    safer_free((void **)&(gvec_elem[eb_indx]));
  }

  safer_free((void **)&gvec_elem);

  safer_free((void **)&rd);
  for (int i = 0; i < num_total_nodes; i++) {
    free(goma_ctx->problem.local_offset[0][i]);
    free(goma_ctx->problem.dolphin[0][i]);
  }
  free(goma_ctx->problem.dolphin[0]);
  free(goma_ctx->problem.local_offset[0]);

  safer_free((void **)&goma_ctx->problem.local_offset);
  safer_free((void **)&goma_ctx->problem.dolphin);

  if (strlen(goma_ctx->soln_out_file)) {
    fclose(file);
  }

  free(gv);
  free(goma_ctx->fem.pg->matrices);

  return;

} /* END of routine hunt_problem  */
/*****************************************************************************/
/*****************************************************************************/
/*  END of file ac_hunt.c  */
/*****************************************************************************/
