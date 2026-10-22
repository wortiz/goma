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
 * Library interface to Goma, see goma.h
 *
 * The stages below are the former body of main(), split so that a host
 * program can drive Goma.  Goma state is still held in global variables, so
 * only one goma_context may exist at a time.
 */

#include <limits.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef HAVE_SUNMATH
#include <sunmath.h>
#endif

#if defined(__linux__)
#ifndef DISABLE_GNU_EXTENSIONS
#define __USE_GNU
#include <fenv.h>
#endif
#endif

#ifdef PARALLEL
#ifdef GOMA_ENABLE_AZTEC
#include "az_aztec.h"
#endif
#endif
extern void handle_ieee(void);

#ifdef USE_CHEMKIN
#include "ck_chemkin_const.h"
#endif
#include "ac_conti.h"
#include "ac_hunt.h"
#include "ac_stability_util.h"
#include "decomp_interface.h"
#include "dp_types.h"
#include "dp_utils.h"
#include "dp_vif.h"
#include "dpi.h"
#include "exo_struct.h"
#ifdef GOMA_ENABLE_METIS
#include "metis_decomp.h"
#endif
#include "brkfix/fix.h"
#include "mm_as.h"
#include "mm_as_alloc.h"
#include "mm_as_structs.h"
#include "mm_eh.h"
#include "mm_elem_block_structs.h"
#include "mm_input.h"
#include "mm_prob_def.h"
#include "rd_dpi.h"
#include "rd_exo.h"
#include "rd_mesh.h"
#include "rf_allo.h"
#include "rf_element_storage_const.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_io_const.h"
#include "rf_io_structs.h"
#include "rf_node_const.h"
#include "rf_pre_proc.h"
#include "rf_solve.h"
#include "rf_solve_segregated.h"
#include "rf_solver.h"
#include "rf_solver_const.h"
#include "std.h"
#include "wr_dpi.h"
#include "wr_exo.h"

#ifdef GOMA_ENABLE_PETSC
#include <petscsys.h>
#endif

/* after the GNU extension include above, for fegetenv/fesetenv */
#include <fenv.h>

#include "goma.h"
#include "goma_reset.h"
#include "goma_reset_check.h"
#include "rf_io.h"
#include "rf_mp.h"
#include "sl_petsc.h"
#include "sl_petsc_complex.h"
#include "sl_petsc_options.h"

/* defined in globals.c */
extern double time_goma_started;
extern char **Argv;
extern int Argc;
extern DDD Noahs_Raven;
extern DDD Noahs_Ark;
extern DDD Noahs_Dove;
extern int unlerr;

#define LAST_LEGAL_STRING "Last Legal String"

static char *legal_notice[] = {
    "/*************************************************************************\n",
    "* Goma - Multiphysics finite element software                            *\n",
    "* Sandia National Laboratories                                           *\n",
    "*                                                                        *\n",
    "* Copyright (c) 2026 Goma Developers, National Technology & Engineering  *\n",
    "*               Solutions of Sandia, LLC (NTESS)                         *\n",
    "*                                                                        *\n",
    "* Under the terms of Contract DE-NA0003525, the U.S. Government retains  *\n",
    "* certain rights in this software.                                       *\n",
    "*                                                                        *\n",
    "* This software is distributed under the GNU General Public License.     *\n",
    "* See LICENSE file.                                                      *\n",
    "\\************************************************************************/\n",
    "\n",
    "PRS, PAS, RRR, KSC, RAC, TAB, DRN, PLH, DAL, IDG, ACK, ACS, RBS, RRL, MMH, SRS\n",
    "HKM, RAR, EDW, PKN, SAR, EMB, KT, DSB, DSH, WWO ...\n",
    "\n",
    LAST_LEGAL_STRING};

enum goma_stage {
  GOMA_STAGE_CREATED,
  GOMA_STAGE_SETUP,
  GOMA_STAGE_SOLVED,
};

struct goma_context {
  int argc;
  char **argv;
  char *yo; /* name of the executable */
  struct Command_line_command **clc;
  int nclc;
  double time_start;
  fenv_t host_fenv; /* floating point environment to restore for the host */
  MPI_Comm comm;    /* duplicate of the host's communicator, goma_comm during the run */
  enum goma_stage stage;
};

static bool goma_library_initialized = false;
static bool goma_context_active = false;
/*
 * Global state is not yet fully reset by goma_destroy, so allow one run per
 * process unless GOMA_EXPERIMENTAL_MULTIPLE_RUNS is set in the environment
 */
static bool goma_context_used = false;

int goma_library_initialize(MPI_Comm comm) {
  if (goma_library_initialized) {
    return -1;
  }
#ifdef GOMA_CHECK_GLOBAL_RESET
  goma_reset_check_snapshot();
#endif
  /* MPI belongs to the host program, never finalize it on a fatal error */
  Goma_Owns_MPI = FALSE;
  goma_library_initialized = true;
  return 0;
}

int goma_library_finalize(void) {
  if (!goma_library_initialized || goma_context_active) {
    return -1;
  }
#ifdef GOMA_ENABLE_PETSC
  goma_petsc_finalize();
#endif
  goma_library_initialized = false;
  return 0;
}

int goma_create(goma_context **ctx_out, MPI_Comm comm, int argc, char **argv) {
  double time_start;
#ifndef PARALLEL
  time_t now;
#endif

  int error;
  int i;
  int j;

  char **ptmp;
  char *yo;

  struct Command_line_command **clc = NULL; /* point to command line structure */
  int nclc = 0;                             /* number of command line commands */

  if (!goma_library_initialized || goma_context_active || ctx_out == NULL) {
    return -1;
  }
  if (goma_context_used && getenv("GOMA_EXPERIMENTAL_MULTIPLE_RUNS") == NULL) {
    return -1;
  }

  goma_context *ctx = calloc(1, sizeof(goma_context));
  if (ctx == NULL) {
    return -1;
  }
  ctx->argc = argc;
  ctx->argv = argv;
#ifdef PARALLEL
  /* Goma communicates over its own duplicate of the host's communicator */
  MPI_Comm_dup(comm, &ctx->comm);
  goma_comm = ctx->comm;
#endif
  goma_context_active = true;
  goma_context_used = true;
  /* Goma may enable floating point exceptions, restored in goma_destroy */
  fegetenv(&ctx->host_fenv);

  yo = argv[0];
  ctx->yo = yo;

#ifdef PARALLEL
  time_start = MPI_Wtime();
#endif /* PARALLEL */
#ifndef PARALLEL
  (void)time(&now);
  time_start = (double)now;
#endif /* PARALLEL */

  time_goma_started = time_start;

  Argv = argv;

  Argc = argc;

#ifdef PARALLEL
  /*
   * Determine the parallel processing status, if any. We need to know
   * pretty early if we're "one of many" or the only process.
   */

  error = MPI_Comm_size(goma_comm, &Num_Proc);
  error = MPI_Comm_rank(goma_comm, &ProcID);

#ifdef GOMA_ENABLE_AZTEC
  /*
   * Setup a default Proc_config so we can use utility routines
   * from Aztec
   */

  AZ_set_proc_config(Proc_Config, goma_comm);
#endif

  /* set the output limit flag if need be */

  if (Num_Proc > DP_PROC_PRINT_LIMIT)
    Unlimited_Output = FALSE;

#ifdef HAVE_MPE_H
  error = MPE_Init_log();
#endif /* HAVE_MPE_H */

  Dim = 0; /* for any hypercube legacy code...  */

#endif /* PARALLEL */

#ifndef PARALLEL
  Dim = 0;
  ProcID = 0;
  Num_Proc = 1;
#endif /* PARALLEL */

  /*
   *   HKM - Change the ieee exception handling based on the machine and
   *         the level of debugging/speed desired. This call currently causes
   *         core dumps for floating point exceptions.
   */

  handle_ieee();

  log_msg("--------------");
  log_msg("GOMA begins...");

  /*
   * Some initial stuff that only the master process does.
   */

  if (ProcID == 0) {
    if (argc > 1) {
      log_msg("Preprocessing command line options.");
      clc = (struct Command_line_command **)smalloc(argc * sizeof(struct Command_line_command *));
      for (i = 0; i < argc; i++) {
        clc[i] = (struct Command_line_command *)smalloc(sizeof(struct Command_line_command));
        clc[i]->type = 0; /* initialize command line structure */
        clc[i]->i_val = 0;
        clc[i]->r_val = 0.;
        clc[i]->string = (char *)smalloc(MAX_COMMAND_LINE_LENGTH * sizeof(char));
        for (j = 0; j < MAX_COMMAND_LINE_LENGTH; j++) {
          clc[i]->string[j] = '\0';
        }
      }
    }

    strcpy(Input_File, "input");
    strcpy(Echo_Input_File, "echo_input");

    if (argc > 1)
      translate_command_line(argc, argv, clc, &nclc);

    ECHO("OPEN", Echo_Input_File);

    echo_command_line(argc, argv, Echo_Input_File);
    print_code_version();
    ptmp = legal_notice;
    while (strcmp(*ptmp, LAST_LEGAL_STRING) != 0) {
      fprintf(stdout, "%s", *ptmp++);
    }
  }

  /*
   *  Allocate the uniform problem description structure and
   *  the problem description structures on all processors
   */
  error = pd_alloc();
  GOMA_EH(error, "pd_alloc problem");

  log_msg("Allocating mp, gn, ...");

  error = mp_alloc();
  GOMA_EH(error, "mp_alloc problem");

  error = gn_alloc();
  GOMA_EH(error, "gn_alloc problem");

  error = ve_alloc();
  GOMA_EH(error, "ve_alloc problem");

  error = elc_alloc();
  GOMA_EH(error, "elc_alloc problem");

  error = elc_rs_alloc();
  GOMA_EH(error, "elc_alloc problem");

  error = cr_alloc();
  GOMA_EH(error, "cr_alloc problem");

  error = evp_alloc();
  GOMA_EH(error, "evp_alloc problem");

  error = tran_alloc();
  GOMA_EH(error, "tran_alloc problem");

  error = eigen_alloc();
  GOMA_EH(error, "eigen_alloc problem");

  error = cont_alloc();
  GOMA_EH(error, "cont_alloc problem");

  error = loca_alloc();
  GOMA_EH(error, "loca_alloc problem");

  error = efv_alloc();
  GOMA_EH(error, "efv_alloc problem");

  /*
   * Read ASCII input file, data files, related exodusII FEM databases.
   */
  if (ProcID == 0) {
    log_msg("Reading input file ...");
    read_input_file(clc, nclc); /* Read ascii input file get file names */

    /* update inputed data to account for command line arguments that
     * might override the input deck...
     */
    log_msg("Overriding any input file specs w/ any command line specs...");
    if (argc > 1)
      apply_command_line(clc, nclc);
  }

  /*
   * The user-defined material properties, etc. available to goma users
   * mean that some dynamically allocated data needs to be communicated.
   *
   * To handle this, sizing information from the input file scan is
   * broadcast in stages so that the other processors can allocate space
   * accordingly to hold the data.
   *
   * Note: instead of handpacking a data structure, use MPI derived datatypes
   * to gather and scatter. Pray this is done efficiently. Certainly it costs
   * less from a memory standpoint.
   */

#ifdef PARALLEL

  /*
   *  Make sure the input file was successully processed before moving on
   */
  check_parallel_error("Input file error");

  /*
   * This is some sizing information that helps fit a little bit more
   * onto the ark later on.
   */

  noahs_raven();

  MPI_Bcast(MPI_BOTTOM, 1, Noahs_Raven->new_type, 0, goma_comm);

#endif

  // We can enable floating point exceptions on all processors here
#if defined(__linux__)
#ifndef DISABLE_GNU_EXTENSIONS
  if (upd->fpe_enabled) {
    feenableexcept((FE_OVERFLOW | FE_DIVBYZERO | FE_INVALID));
  }
#endif
#endif

#ifdef PARALLEL
  /*
   * Get the other processors ready to handle ark data.
   */

  raven_landing();

  /*
   * This is the main body of communicated information, including some
   * whose sizes were determined because of advanced legwork by the raven.
   */

  noahs_ark();
  MPI_Bcast(MPI_BOTTOM, 1, Noahs_Ark->new_type, 0, goma_comm);

  /*
   * Chemkin was initialized on processor zero during the input file
   * process. Now, distribute it to all processors
   */
#ifdef USE_CHEMKIN
  if (Chemkin_Needed) {
    chemkin_initialize_mp();
  }
#endif

  /*
   * Once the ark has landed, there are additional things that will need to
   * be sent by dove. Example: BC_Types[]->u-BC arrays.
   *
   */

  ark_landing();

  noahs_dove();
  MPI_Bcast(MPI_BOTTOM, 1, Noahs_Dove->new_type, 0, goma_comm);

#endif /* End of ifdef PARALLEL */

  if (efv->Num_external_field != 0) {
    for (i = 0; i < efv->Num_external_field; i++) {
      memcpy(efv->file_nm_serial[i], efv->file_nm[i], MAX_FNL);
    }
  }

  ctx->time_start = time_start;
  ctx->clc = clc;
  ctx->nclc = nclc;
  ctx->stage = GOMA_STAGE_CREATED;
  *ctx_out = ctx;
  return 0;
}

int goma_setup(goma_context *ctx) {
  int error;
  int i;

  if (ctx == NULL || ctx->stage != GOMA_STAGE_CREATED) {
    return -1;
  }
  char *yo = ctx->yo;

#ifdef GOMA_ENABLE_METIS
  /* Now break the exodus files */
  if (Decompose_Flag == 1 && Num_Proc > 1 && ProcID == 0) {
    char **filenames = malloc(sizeof(char *) * (2 + MAX_EXTERNAL_FIELD));
    int n_files = 1;
    filenames[0] = malloc(sizeof(char) * MAX_FNL);
    memcpy(filenames[0], ExoFile, MAX_FNL);
    if (strcmp(ExoAuxFile, "") != 0) {
      filenames[n_files] = malloc(sizeof(char) * MAX_FNL);
      memcpy(filenames[n_files], ExoAuxFile, MAX_FNL);
      n_files++;
    }

    if (efv->Num_external_field != 0) {
      for (i = 0; i < efv->Num_external_field; i++) {
        if (efv->ipix[i] == 1 || efv->ipix[i] == 2)
          continue;
        filenames[n_files] = malloc(sizeof(char) * MAX_FNL);
        memcpy(filenames[n_files], efv->file_nm[i], MAX_FNL);
        n_files++;
      }
    }

    goma_metis_decomposition(filenames, n_files);

    for (int i = 0; i < n_files; i++) {
      free(filenames[i]);
    }
    free(filenames);
  }
  check_parallel_error("Error in decomposing exodus files");
#endif
  MPI_Barrier(goma_comm);

  /*
   * For parallel execution, assume the following variables will be changed
   * to reflect the multiple file aspect of the problem.
   *
   *	FEM file = file.exoII		--> file_3of15.exoII
   *
   *	Output EXODUS II file = out.exoII --> out_3of15.exoII
   *
   */

  /*
   * Allocate space for structures holding the EXODUS II finite element
   * database information and for the Distributed Processing information.
   *
   * These are mostly skeletons with pointers that get allocated in the
   * rd_exoII and rd_dpi routines. Remember to free up those arrays first
   * before freeing the major pointers.
   */

  EXO_ptr = alloc_struct_1(Exo_DB, 1);
  init_exo_struct(EXO_ptr);
  DPI_ptr = alloc_struct_1(Dpi, 1);
  init_dpi_struct(DPI_ptr);

  log_msg("Reading mesh from EXODUS II file...");
  error = read_mesh_exoII(EXO_ptr, DPI_ptr);

  /*
   *   Missing files on any processor are detected at a lower level
   *   forcing a return to the higher level
   *         rd_exo -->  rd_mesh  -->  main
   *   Shutdown now, if any of the exodus files weren't found
   */
  if (error < 0) {
    return (-1);
  }

  /*
   * All of the MPI_Type_commit() calls called behind the scenes that build
   * the dove, ark and raven really allocated memory. Let's free it up now that
   * the initial information has been communicated.
   */

#ifdef PARALLEL
  MPI_Type_free(&(Noahs_Raven->new_type));
  MPI_Type_free(&(Noahs_Ark->new_type));
  MPI_Type_free(&(Noahs_Dove->new_type));
#endif

  /*
   * Setup the rest of the Problem Description structure that depends on
   * the mesh that was read in from the EXODUS II file...
   *
   * Note that memory allocation and some setup has already been performed
   * in mm_input()...
   */

  error = setup_pd();
  GOMA_EH(error, "Problem setting up Problem_Description.");
  /*
   * Let's check to see if we need the large elasto-plastic global tensors
   * and allocate them if so
   */
  error = evp_tensor_alloc(EXO_ptr);
  GOMA_EH(error, "Problems setting up evp tensors");

  /*
   * Now that we know about what kind of problem we're solving and the
   * mesh information, let's allocate space for elemental assembly structures
   *
   */
  log_msg("Assembly allocation...");

  error = assembly_alloc(EXO_ptr);
  GOMA_EH(error, "Problem from assembly_alloc");

  if (Debug_Flag) {
    DPRINTF(stderr, "%s:  setting up EXODUS II output files...\n", yo);
  }

  /*
   * These are not critical - just niceties. Also, they should not overburden
   * your db with too much of this - they're capped verbiage compliant routines.
   */

  add_qa_stamp(EXO_ptr);

  add_info_stamp(EXO_ptr);

  /*
   * If the output EXODUS II database file is different from the input
   * file, then we'll need to replicate all the basic mesh information.
   * But, remember that if we're parallel, that the output file names must
   * be multiplexed first...
   */
  if (Num_Proc > 1) {
    multiname(ExoFileOut, ProcID, Num_Proc);
    multiname(Init_GuessFile, ProcID, Num_Proc);

    if (strcmp(Soln_OutFile, "") != 0) {
      multiname(Soln_OutFile, ProcID, Num_Proc);
    }

    if (strcmp(ExoAuxFile, "") != 0) {
      multiname(ExoAuxFile, ProcID, Num_Proc);
    }

    if (efv->Num_external_field != 0) {
      for (i = 0; i < efv->Num_external_field; i++) {
        multiname(efv->file_nm[i], ProcID, Num_Proc);
      }
    }
  }

  /***********************************************************************/
  /***********************************************************************/
  /***********************************************************************/
  /*
   *   Preprocess the exodus mesh
   *        -> Allocate pointers to structures containing element
   *           side bc info, First_Elem_Side_BC_Array, and
   *           element edge info, First_Elem_Edge_BC_Array.
   *        -> Determine Unique_Element_Types[] array
   */
  log_msg("Pre processing of mesh...");
#ifdef PARALLEL
  error = MPI_Barrier(goma_comm);
#endif
  pre_process(EXO_ptr);

  /***********************************************************************/
  /***********************************************************************/
  /***********************************************************************/
  /*
   * Load up a few key indeces in the bfd prototype basis function structures
   * and make sure that each active eqn/vbl has a bf[v] that points to the
   * right bfd[]...needs pre_process to find out the number of unique
   * element types in the problem.
   */

  log_msg("Basis function initialization...");
  error = bf_init(EXO_ptr);
  GOMA_EH(error, "Problem from bf_init");

  /*
   * check for parallel errors before continuing
   */
  check_parallel_error("Error encountered in problem setup");

  /***********************************************************************/
  /***********************************************************************/
  /***********************************************************************/
  /*
   * Allocate space for each communication exchange description.
   */
#ifdef PARALLEL
  cx = calloc(upd->Total_Num_Matrices, sizeof(Comm_Ex *));
  if (DPI_ptr->num_neighbors > 0) {

    int imtrx;
    for (imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
      cx[imtrx] = alloc_struct_1(Comm_Ex, DPI_ptr->num_neighbors);
      Request = alloc_struct_1(MPI_Request, Num_Requests * DPI_ptr->num_neighbors);
      Status = alloc_struct_1(MPI_Status, Num_Requests * DPI_ptr->num_neighbors);
    }
  }
#endif

  /***********************************************************************/
  /***********************************************************************/
  /***********************************************************************/
  /*
   *                           SET UP THE PROBLEM
   *
   * Setup node-based structures
   * Finalise how boundary conditions are to be handled
   * Determine what unknowns are at each owned node and then tell
   *  neighboring processors about your nodes
   * Set up communications pattern for fast unknown updates between
   *  processors.
   */
  (void)setup_problem(EXO_ptr, DPI_ptr);

  /*
   * check for parallel errors before continuing
   */
  check_parallel_error("Error encountered in problem setup");
  /***********************************************************************/
  /***********************************************************************/
  /***********************************************************************/
  /*
   *                     WRITE OUT INITIAL INFO TO EXODUS FILE
   */

  /*
   *  Only have to initialize the exodus file if we are using different
   *  files for the output versus the input mesh
   */
  if (strcmp(ExoFile, ExoFileOut)) {
    /*
     * Temporarily we'll need to renumber the nodes and elements in the
     * mesh to be 1-based. After writing, return to the 0 based indexing
     * that is more convenient in C.
     */
    one_base(EXO_ptr, Num_Proc);
    wr_mesh_exo(EXO_ptr, ExoFileOut, 0);
    zero_base(EXO_ptr);

    /*
     * If running on a distributed computer, augment the plain finite
     * element information of EXODUS with the description of how this
     * piece fits into the global problem.
     */
    if (Num_Proc > 1) {
      wr_dpi(DPI_ptr, ExoFileOut);
    }
  }
#ifdef FP_EXCEPT
  feenableexcept((FE_OVERFLOW | FE_DIVBYZERO | FE_INVALID));
#endif

  ctx->stage = GOMA_STAGE_SETUP;
  return 0;
}

int goma_solve(goma_context *ctx) {
  int error = 0;

  if (ctx == NULL || ctx->stage != GOMA_STAGE_SETUP) {
    return -1;
  }
  char *yo = ctx->yo;

  /***********************************************************************/
  /***********************************************************************/
  /***********************************************************************/
  /*
   *                           SOLVE THE PROBLEM
   */

  if (upd->Total_Num_Matrices == 1) {
    pg->imtrx = 0;

    if (Debug_Flag) {
      switch (Continuation) {
      case ALC_ZEROTH:
        P0PRINTF("%s: continue_problem (zeroth order) ...\n", yo);
        break;
      case ALC_FIRST:
        P0PRINTF("%s: continue_problem (first order) ...\n", yo);
        break;
      case HUN_ZEROTH:
        P0PRINTF("%s: hunt_problem (zeroth order) ...\n", yo);
        break;
      case HUN_FIRST:
        P0PRINTF("%s: hunt_problem (first order) ...\n", yo);
        break;
      case LOCA:
        P0PRINTF("%s: do_loca ...\n", yo);
        break;
      default:
        P0PRINTF("%s: solve_problem...\n", yo);
        break;
      }
    }

    if (TimeIntegration == TRANSIENT) {
      Continuation = ALC_NONE;
      if (Debug_Flag) {
        P0PRINTF("%s: solve_problem...TRANSIENT superceded Continuation...\n", yo);
      }
      solve_problem(EXO_ptr, DPI_ptr, NULL);
    }

    switch (Continuation) {
    case ALC_ZEROTH:
    case ALC_FIRST:
      log_msg("Solving continuation problem");
      continue_problem(cx[0], EXO_ptr, DPI_ptr);
      break;
    case HUN_ZEROTH:
    case HUN_FIRST:
      log_msg("Solving hunt problem");
      hunt_problem(cx[0], EXO_ptr, DPI_ptr);
      break;
    case LOCA:
      log_msg("Solving continuation problem with LOCA");
      error = do_loca(cx[0], EXO_ptr, DPI_ptr);
      break;
    default:
      log_msg("Solving problem");
      if (loca_in->Cont_Alg == LOCA_LSA_ONLY) {
        error = do_loca(cx[0], EXO_ptr, DPI_ptr);
      } else if (TimeIntegration != TRANSIENT) {
        solve_problem(EXO_ptr, DPI_ptr, NULL);
      }
      break;
    }
  } else { /* End of if upd->Total_Num_Matrices === 1 */
    solve_problem_segregated(EXO_ptr, DPI_ptr, NULL);
  }

#ifdef PARALLEL
  MPI_Barrier(goma_comm);
#endif

  fix_output();

  /***********************************************************************/
  /***********************************************************************/
  /***********************************************************************/
  /*
   *  PRINT A MESSAGE TO STDOUT SAYING WE ARE DONE
   */
  P0PRINTF("\n-done\n\n");

  ctx->stage = GOMA_STAGE_SOLVED;
  /* do_loca returns the number of continuation steps on success */
  return error < 0 ? error : 0;
}

int goma_destroy(goma_context **ctx_ptr) {
  double total_time;
#ifndef PARALLEL
  time_t now;
#endif
  int i;

  if (ctx_ptr == NULL || *ctx_ptr == NULL) {
    return -1;
  }
  goma_context *ctx = *ctx_ptr;
#ifdef GOMA_CHECK_GLOBAL_RESET
  /* ProcID is reset below */
  int check_rank = ProcID;
#endif
  struct Command_line_command **clc = ctx->clc;
  char *yo = ctx->yo;
  double time_start = ctx->time_start;

  /***********************************************************************/
  /***********************************************************************/
  /***********************************************************************/
  /*
   *       FREE MEMORY ALLOCATED BY THE PROGRAM
   */
  /* the mesh and problem structures exist once goma_setup has finished */
  if (ctx->stage != GOMA_STAGE_CREATED) {
    /*
     * free the element block / element based structures
     */
    free_element_blocks(EXO_ptr);

    /* the duplicated boundary condition lists are sized from Nodes */
    bc_conflict_free_globals();

    /*
     * free nodal based structures
     */
    free_nodes();
    free_problem(EXO_ptr, DPI_ptr);
  }

  /*
   * Free command line stuff
   */
  if (ProcID == 0) {
    if (ctx->argc > 1) {
      for (i = 0; i < ctx->argc; i++) {
        safer_free((void **)&(clc[i]->string));
        safer_free((void **)(clc + i));
      }
      safer_free((void **)&clc);
      ctx->clc = NULL;
    }
  }

  /*
   * Free exodus database structures
   */
  if (EXO_ptr != NULL) {
    free_exo(EXO_ptr);
    safer_free((void **)&EXO_ptr);
  }

  if (DPI_ptr != NULL) {
    if (Num_Proc > 1) {
      free_dpi(DPI_ptr);
    } else {
      free_dpi_uni(DPI_ptr);
    }
    safer_free((void **)&DPI_ptr);
  }

  /*
   * Remove front scratch file [/tmp/lu.'pid'.0]
   */
  if (Linear_Solver == FRONT) {
    unlerr = unlink(front_scratch_directory);
    GOMA_WH(unlerr, "Unlink problem with front scratch file");
  }

#ifdef PARALLEL
  total_time = (MPI_Wtime() - time_start) / 60.;
  DPRINTF(stdout, "\nProc 0 runtime: %10.2f Minutes.\n\n", total_time);
#endif
#ifndef PARALLEL
  (void)time(&now);
  total_time = (double)(now)-time_start;
  fprintf(stdout, "\nProc 0 runtime: %10.2f Minutes.\n\n", total_time / 60);
#endif
  fflush(stdout);
  fflush(stderr);
  log_msg("GOMA ends normally.");
  free(Status);
  free(Request);
  Status = NULL;
  Request = NULL;

#ifdef GOMA_ENABLE_PETSC
  goma_petsc_reset_options();
#endif
  fesetenv(&ctx->host_fenv);

  /*
   * Reset global state so Goma can run again, in reverse order of setup.
   * as_free_globals frees upd, pd_glob, ... so it must be last.
   */
#if defined(GOMA_ENABLE_PETSC) && !(PETSC_USE_COMPLEX)
  goma_petsc_free_post_proc_matrix();
#endif
  vif_free_globals();
  assembly_free();
  rf_solve_free_globals();
  rf_solve_segregated_free_globals();
  sol_nonlinear_free_globals();
  umf_free_globals();
  lu_free_globals();
  luf_free_globals();
  fill_solid_free_globals();
  fill_porous_free_globals();
  sl_aux_free_globals();
  rotate_free_globals();
  rotate_coordinates_free_globals();
  shell_util_free_globals();
  fill_species_free_globals();
  fill_ls_free_globals();
  loca_interface_free_globals();
  loca_util_free_globals();
  loca_bord_free_globals();
  amesos_free_globals();
  leastsquares_free_globals();
  stability_free_globals();
  side_data_free_globals();
  pixel_image_free_globals();
  el_quality_free_globals();
  particles_free_globals();
  eggrollwrap_free_globals();
  omega_h_free_globals();
  ad_util_free_globals();
  stratimikos_free_globals();
  post_proc_free_globals();
  fill_free_globals();
  fill_ptrs_free_globals();
  fill_util_free_globals();
  fill_stabilization_free_globals();
  qtensor_free_globals();
  bc_free_globals();

  unknown_map_free_globals();
  vars_free_globals();
  map_comm_vec_free_globals();
  rd_mesh_free_globals();
  sl_util_free_globals();
  dp_utils_free_globals();
  input_free_globals();
  globals_free_globals();
  eh_free_globals();
  as_free_globals();

#ifdef PARALLEL
  /* goma_comm was reset by eh_free_globals */
  MPI_Comm_free(&ctx->comm);
#endif
  free(ctx);
  *ctx_ptr = NULL;
  goma_context_active = false;

#ifdef GOMA_CHECK_GLOBAL_RESET
  char check_file[MAX_FNL];
  snprintf(check_file, MAX_FNL, "goma_reset_check.%d.txt", check_rank);
  FILE *check_out = fopen(check_file, "w");
  if (check_out != NULL) {
    int num_ranges = goma_reset_check_compare(check_out);
    fclose(check_out);
    fprintf(stderr, "P_%d goma reset check: %d differing ranges written to %s\n", check_rank,
            num_ranges, check_file);
  }
#endif
  return 0;
}

/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

void print_code_version(void)

/*
 * Print the code version to standard out
 */
{
  printf("%s\n", GOMA_VERSION);
}

void echo_command_line(int argc, char *argv[], char *echo_file) {
  int istr = 0;
  char *echo_string = malloc(sizeof(char) * 65536);
  echo_string[0] = '\0';
  time_t start_time;
  char *time_string;

  start_time = time(NULL);
  time_string = asctime(localtime(&start_time));

  SPF(echo_string, "Command line :");

  while (istr < argc) {
    SPF(endofstring(echo_string), " %s", argv[istr]);
    istr++;
  }

  ECHO(echo_string, echo_file);

  SPF(echo_string, "Run Time : %s", time_string);

  SPF(endofstring(echo_string), "Version : %s", GOMA_VERSION);

  ECHO(echo_string, echo_file);
  free(echo_string);
}
