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

#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef HAVE_SUNMATH
#include <sunmath.h>
#endif

#ifdef FP_EXCEPT
#define __USE_GNU
#include <fenv.h>
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

static void print_fix_version(void);

static void print_usage(void) {
  DPRINTF(stdout, "usage: fix <exodus_output>.<numproc>.<single>\n\nexample: fix out.exoII.24.00");
  DPRINTF(stdout, "    example: fix out.exoII.24.00\n\n");
}

static void get_fix_info(char *filename, int *num_procs, char mono_name[MAX_FNL]) {

  char infile[MAX_FNL];
  strncpy(infile, filename, MAX_FNL - 1);
  char *tokens[100];

  char previous_token[MAX_FNL];
  char last_token[MAX_FNL];

  mono_name[0] = '\0';
  previous_token[0] = '\0';
  last_token[0] = '\0';

  char *parse_string = infile;

  char *token = strtok(parse_string, ".");
  int tk = 0;

  strncpy(last_token, token, MAX_FNL - 1);
  while (token) {
    GOMA_ASSERT_ALWAYS(tk < 100);
    tokens[tk++] = token;
    snprintf(previous_token, MAX_FNL, "%s", last_token);
    snprintf(last_token, MAX_FNL, "%s", token);
    token = strtok(NULL, ".");
  }

  if (tk < 4) {
    GOMA_EH(GOMA_ERROR, "Expected to find a file with at least 4 parts, got %d from %s", tk,
            filename);
  }

  for (int i = 0; i < (tk - 2); i++) {
    strcat(mono_name, tokens[i]);
    if (i < (tk - 3)) {
      strcat(mono_name, ".");
    }
  }

  char *p_end;
  long num_proc_parse = strtol(previous_token, &p_end, 10);
  if (errno == ERANGE) {
    GOMA_EH(GOMA_ERROR, "Could not parse number of processors %s from %s", previous_token,
            filename);
  }

  *num_procs = num_proc_parse;
}

int main(int argc, char **argv) {
  dbl time_start, total_time;

#ifdef PARALLEL
  MPI_Init(&argc, &argv);
  time_start = MPI_Wtime();
#endif /* PARALLEL */

  int rank;
  int size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  ProcID = rank;
  Num_Proc = size;

  if (size > 1) {
    GOMA_EH(GOMA_ERROR, "Fix only supports running on one processor.");
  }

  if (rank == 0) {

    print_fix_version();
    if (argc != 2) {
      print_usage();
      GOMA_EH(GOMA_ERROR, "Unknown number of arguments %d", argc - 1);
    }

    if (argc == 2) {
      if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_usage();
        goto fix_exit;
      }
    }

    int num_procs;
    char mono_name[MAX_FNL];

    get_fix_info(argv[1], &num_procs, mono_name);

    Num_Proc = num_procs;

    DPRINTF(stdout, "Fixing file %s with %d procs\n", mono_name, num_procs);
    fix_exo_file(num_procs, mono_name);
  }
  total_time = (MPI_Wtime() - time_start) / 60.;
  DPRINTF(stdout, "\nProc 0 runtime: %10.2f Minutes.\n\n", total_time);
fix_exit:
#ifdef PARALLEL
  MPI_Finalize();
#endif
  fflush(stdout);
  fflush(stderr);
  return (0);
}

/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

static void print_fix_version(void)

/*
 * Print the code version to standard out
 */
{
  printf("fix (Goma %s)\n", GOMA_VERSION);
}

/***********************************************************************/
/***********************************************************************/
/***********************************************************************/
