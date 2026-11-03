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

#ifndef GOMA_UTIL_GOMA_CONTEXT_H
#define GOMA_UTIL_GOMA_CONTEXT_H

#include <mpi.h>

#include "rf_io_const.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * GomaContext collects process-wide state that used to live in individual
 * global variables: MPI/parallel information and I/O / run configuration.
 *
 * A single instance is owned by goma_context.c and reachable through
 * goma_ctx, which is never NULL.
 */
typedef struct GomaContext {
  /* Parallel machine description and MPI communication bookkeeping */
  int num_proc;               /* Total number of processors */
  int proc_id;                /* This processor's number */
  int dim;                    /* Dimension of logical hypercube */
  MPI_Request *request;       /* Nonblocking send/recv requests */
  MPI_Status *status;         /* Nonblocking send/recv statuses */
  int num_requests;           /* Number of entries in request/status */
  int unlimited_output;       /* flag to unlimit multiprocessor output */

  /* Input / output files */
  char input_file[MAX_FNL];        /* input EXODUS II database w/ problem defn */
  char exo_file[MAX_FNL];          /* input EXODUS II database w/ problem defn */
  char exo_lb_file[MAX_FNL];       /* EXODUS II load balance info for mesh */
  char exo_file_out[MAX_FNL];      /* output EXODUS II database w/ results */
  char exo_file_out_mono[MAX_FNL]; /* output EXODUS II database without per proc string */
  char exo_aux_file[MAX_FNL];      /* auxiliary EXODUS II database for initguess */
  char echo_input_file[MAX_FNL];   /* echo of problem def file */
  char domain_mapping_file[MAX_FNL]; /* Maps materials and material boundary names into chemkin
                                        domains and surface/volumetric domains */
  char init_guess_file[MAX_FNL];   /* ASCII file holding initial guess */
  char soln_out_file[MAX_FNL];     /* ASCII file holding solution vector */
  int exo_time_plane;              /* Time plane # or continuation # of soln for initial guess */
  int cpu_word_size;               /* Precision variables for exodus II files */
  int io_word_size;

  /* Run configuration */
  int decompose_flag;              /* Flag to check for built-in brking */
  int decompose_type;
  int skip_fix;
  int debug_flag;                  /* Level of debug info to stdout: 0 none, 1 minimum, 2 medium,
                                      3 maximum, -1 check jacobian, -2 check jacobian w/ scaling */
  int iout;                        /* Level of diagnostic output for the program */
  int new_parser_flag;             /* 0: old parser, 1: flex/bison parser */
  int write_intermediate_solutions; /* Write solution data at each Newton iteration */
  int write_initial_solution;       /* Write initial solution to ascii and exodus output */
} GomaContext;

/* Set every field of ctx to the Goma startup default. */
void goma_context_init(GomaContext *ctx);

/* The process-wide context. Never NULL. */
extern GomaContext *goma_ctx;

#ifdef __cplusplus
}
#endif

#endif /* GOMA_UTIL_GOMA_CONTEXT_H */
