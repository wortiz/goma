/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) Goma Developers, National Technology & Engineering        *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

/*
 * Runs Goma several times in one process through the goma.h interface
 *
 * usage: goma_lib_sequence [--ranks=<n>] <dir> [--stop-after=create|setup]
 *                          [--advance-steps=<n> | --advance-to=<dt>]
 *                          [--solve-after=<n>] [--import-value-step=<step>]
 *                          [--check-imports=<file>]
 *                          [--run-ranks=<first>:<n>] [--expect-error]
 *                          [--log=<file>] <goma args>
 *                          [-- <dir> ...] ...
 *
 * Each run changes to <dir> and runs Goma with the given arguments, for
 * example:
 *
 *   goma_lib_sequence problemA -i input -- problemB -i input
 *
 * --stop-after destroys the context after goma_create or goma_setup instead
 * of solving, to test teardown of a partly set up problem.
 *
 * --advance-steps solves with goma_advance_steps(<n>) in a loop and
 * --advance-to with goma_advance_to() at every <dt> of simulation time,
 * checking that each call lands on its target, instead of goma_solve.
 * --check-fields compares goma_get_field/goma_get_post_field after each of
 * these calls with every nodal variable of the last time step in output
 * file <file> when that step is the current time (with .<np>.<rank> added
 * in parallel).
 *
 * --import=<name>:<file> (repeatable) sets imported external field <name>
 * (External Field = <name> ... IMPORT in the deck) with goma_set_external_field
 * to the nodal variable <name> of the last time step in mesh file <file>,
 * matched to Goma's nodes by global node id.  With --advance-*, it is set again
 * before each call; --feedback=<name>:<field>:<scale> then adds <scale> times
 * Goma's current <field> (goma_get_field), like a coupled code would.
 * --import-value=<name>:<value> (repeatable) sets imported nodal field <name>
 * to <value> at every node once, before the problem is solved or advanced, so
 * that it keeps those values (as transferred by any mesh adaptation) from then
 * on; with --advance-*, goma_get_post_field must return <value> at every owned
 * node after each call.
 * --import-value-step=<step> (with --advance-*) sets the --import-value fields
 * again before each call after the first, to <value> + <k> * <step> before
 * call <k>, sized for the mesh as it is then (adapted by earlier calls or
 * not), and they must have these values after the call.
 * --solve-after=<n> (with --advance-*) finishes with goma_solve after <n>
 * calls, setting the imported fields first as before a call.
 * --check-imports=<file> checks that the --import-value fields at the last
 * time step of output file <file> (with .<np>.<rank> added in parallel) have
 * the values last set.
 * --declare=<name>:<interpolation> (repeatable) declares the imported nodal
 * field <name> with goma_declare_external_field before goma_create, in place
 * of an External Field card.
 *
 * --log writes
 * the standard output of rank 0 during the run to <file> (relative to the
 * starting directory).
 *
 * --ranks runs Goma on a communicator split from MPI_COMM_WORLD that holds
 * only its first <n> ranks, the other ranks wait, to check that Goma does not
 * communicate outside the communicator it is given.
 *
 * --expect-setup-error checks a rejected setup, failed solve/advance/retry calls,
 * and teardown, allowing the next run to verify recovery in the same process.
 *
 * --expect-error expects goma_solve or goma_advance_* to return an error, for
 * a problem whose solver gives up; the run fails if they do not.
 *
 * --run-ranks runs one problem on ranks <first> .. <first>+<n>-1 of the
 * library's communicator (all of it by default), the other ranks skip it, so
 * that runs in one process can use different sets of ranks.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* RTLD_DEFAULT */
#endif
#include <mpi.h>
#include <stdio.h>
#ifdef __GLIBC__
#include <malloc.h>
#endif
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "exodusII.h"
#include "goma.h"

enum stop_after { STOP_AFTER_CREATE, STOP_AFTER_SETUP, STOP_AFTER_SOLVE };

struct solve_mode {
  int advance_steps;        /* > 0: goma_advance_steps(advance_steps) */
  double advance_to;        /* > 0: goma_advance_to() at multiples of advance_to */
  const char *check_fields; /* output file to compare the fields with */
  int num_imports;          /* --import=<name>:<file> options */
  const char **imports;
  const char *feedback; /* --feedback=<name>:<field>:<scale> */
  int num_declares;     /* --declare=<name>:<interpolation> options */
  const char **declares;
  int num_import_values; /* --import-value=<name>:<value> options */
  const char **import_values;
  double import_value_step;  /* --import-value-step=<step> */
  int solve_after;           /* > 0: goma_solve after this many calls */
  const char *check_imports; /* output file to check the --import-value fields in */
  int expect_setup_error;    /* assert that a failed setup cannot be solved or retried */
};

#define MAX_IMPORTS 8

/* Set imported external field name to its values in mesh file file, plus scale times
 * Goma's field feedback if feedback is not NULL */
static int import_field(
    goma_context *ctx, const char *name, const char *file, const char *feedback, double scale) {
  int cpu_ws = 8, io_ws = 0;
  float version;
  int exoid = ex_open(file, EX_READ, &cpu_ws, &io_ws, &version);
  if (exoid < 0) {
    fprintf(stderr, "goma_lib_sequence: could not open %s\n", file);
    return -1;
  }
  int num_steps = ex_inquire_int(exoid, EX_INQ_TIME);
  int num_file_nodes = ex_inquire_int(exoid, EX_INQ_NODES);
  int num_vars = 0, var = 0;
  ex_get_variable_param(exoid, EX_NODAL, &num_vars);
  for (int v = 1; v <= num_vars; v++) {
    char var_name[MAX_STR_LENGTH + 1];
    ex_get_variable_name(exoid, EX_NODAL, v, var_name);
    if (strcmp(var_name, name) == 0) {
      var = v;
    }
  }
  if (num_steps < 1 || var == 0) {
    fprintf(stderr, "goma_lib_sequence: no variable %s in %s\n", name, file);
    ex_close(exoid);
    return -1;
  }
  double *file_values = malloc(sizeof(double) * num_file_nodes);
  int *file_ids = malloc(sizeof(int) * num_file_nodes);
  ex_get_var(exoid, num_steps, EX_NODAL, var, 1, num_file_nodes, file_values);
  ex_get_id_map(exoid, EX_NODE_MAP, file_ids);
  ex_close(exoid);

  /* the file is the undecomposed mesh, its node ids are the global ids */
  long long max_id = 0;
  for (int i = 0; i < num_file_nodes; i++) {
    max_id = file_ids[i] > max_id ? file_ids[i] : max_id;
  }
  double *by_id = calloc(max_id + 1, sizeof(double));
  for (int i = 0; i < num_file_nodes; i++) {
    by_id[file_ids[i]] = file_values[i];
  }
  goma_mesh_info mi;
  goma_mesh_get_info(ctx, &mi);
  long long *ids = malloc(sizeof(long long) * mi.num_nodes);
  double *values = malloc(sizeof(double) * mi.num_nodes);
  goma_mesh_get_node_global_ids(ctx, ids);
  for (int i = 0; i < mi.num_nodes; i++) {
    values[i] = ids[i] <= max_id ? by_id[ids[i]] : 0.0;
  }
  int error = 0;
  if (feedback != NULL) {
    double *field = malloc(sizeof(double) * mi.num_nodes);
    error = goma_get_field(ctx, feedback, 0, field);
    for (int i = 0; i < mi.num_nodes && error == 0; i++) {
      values[i] += scale * field[i];
    }
    free(field);
  }
  if (error == 0) {
    error = goma_set_external_field(ctx, name, GOMA_NODAL, values);
  }
  free(file_values);
  free(file_ids);
  free(by_id);
  free(ids);
  free(values);
  return error;
}

/* Compare the fields Goma returns with the output file when its last step is
 * the current time, returns the number of differing values or -1 */
static int check_fields(goma_context *ctx, MPI_Comm comm, const char *file, int *checked) {
  int rank, np;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  char path[4096];
  if (np > 1) {
    snprintf(path, sizeof(path), "%s.%d.%d", file, np, rank);
  } else {
    snprintf(path, sizeof(path), "%s", file);
  }
  double t, dt;
  goma_mesh_info mi;
  if (goma_get_time(ctx, &t, &dt) != 0 || goma_mesh_get_info(ctx, &mi) != 0) {
    return -1;
  }
  int cpu_ws = 8, io_ws = 0;
  float version;
  int exoid = ex_open(path, EX_READ, &cpu_ws, &io_ws, &version);
  int num_steps = exoid < 0 ? 0 : ex_inquire_int(exoid, EX_INQ_TIME);
  double t_file = 0.0;
  int num_vars = 0;
  if (num_steps > 0) {
    ex_get_time(exoid, num_steps, &t_file);
  }
  /* a parallel output file holds the owned nodes, numbered first */
  int num_nodes = exoid < 0 ? 0 : ex_inquire_int(exoid, EX_INQ_NODES);
  /* all ranks must agree, goma_get_post_field is collective */
  /* a steady problem that writes its Newton iterations numbers them as times */
  int steady = t == 0.0 && dt == 0.0;
  int here = num_steps > 0 && (t_file == t || steady) &&
             (num_nodes == mi.num_nodes || num_nodes == mi.num_owned_nodes);
  int all;
  MPI_Allreduce(&here, &all, 1, MPI_INT, MPI_MIN, comm);
  if (all) {
    ex_get_variable_param(exoid, EX_NODAL, &num_vars);
  }
  int bad = 0;
  double *expected = malloc(sizeof(double) * (mi.num_nodes + 1));
  double *values = malloc(sizeof(double) * (mi.num_nodes + 1));
  /* the file may order the nodes differently, match them by global id */
  int *local = malloc(sizeof(int) * (mi.num_nodes + 1));
  if (num_vars > 0) {
    long long *ids = malloc(sizeof(long long) * mi.num_nodes);
    int *file_ids = malloc(sizeof(int) * (num_nodes + 1));
    goma_mesh_get_node_global_ids(ctx, ids);
    ex_get_id_map(exoid, EX_NODE_MAP, file_ids);
    for (int i = 0; i < num_nodes; i++) {
      local[i] = -1;
      for (int j = 0; j < mi.num_nodes && local[i] < 0; j++) {
        if (ids[j] == file_ids[i]) {
          local[i] = j;
        }
      }
      if (local[i] < 0) {
        fprintf(stderr, "goma_lib_sequence: rank %d node id %d of %s not found\n", rank,
                file_ids[i], path);
        bad++;
        local[i] = 0;
      }
    }
    free(ids);
    free(file_ids);
  }
  for (int v = 1; v <= num_vars; v++) {
    char name[MAX_STR_LENGTH + 1];
    ex_get_variable_name(exoid, EX_NODAL, v, name);
    ex_get_var(exoid, num_steps, EX_NODAL, v, 1, num_nodes, expected);
    /* post processing names first: PRESSURE, for example, is a post processing variable
     * in the file, but goma_get_field takes it as the variable name of P */
    int error = goma_get_post_field(ctx, name, values);
    if (error != 0) {
      error = goma_get_field(ctx, name, 0, values);
    }
    if (error != 0) {
      /* time derivatives, for example, are not available */
      continue;
    }
    (*checked)++;
    int differ = 0;
    for (int i = 0; i < num_nodes; i++) {
      differ += values[local[i]] != expected[i];
    }
    if (differ > 0) {
      fprintf(stderr, "goma_lib_sequence: rank %d %s at t=%g differs at %d of %d nodes\n", rank,
              name, t, differ, num_nodes);
    }
    bad += differ;
  }
  free(expected);
  free(values);
  free(local);
  if (exoid >= 0) {
    ex_close(exoid);
  }
  return exoid < 0 ? -1 : bad;
}

/* Name and value before call k of --import-value field i, returns -1 if it is malformed */
static int import_value(struct solve_mode mode, int i, int k, char name[256], double *value) {
  if (sscanf(mode.import_values[i], "%255[^:]:%lf", name, value) != 2) {
    return -1;
  }
  *value += k * mode.import_value_step;
  return 0;
}

/* Set the imported external fields before call k (0: before the problem starts), see
 * import_field() */
static int set_imports(goma_context *ctx, struct solve_mode mode, int k) {
  int error = 0;
  int started = k > 0;
  int set_values = !started || mode.import_value_step != 0.0;
  for (int i = 0; i < mode.num_import_values && set_values && error == 0; i++) {
    char name[256];
    double value;
    goma_mesh_info info;
    if (import_value(mode, i, k, name, &value) != 0 || goma_mesh_get_info(ctx, &info) != 0) {
      return -1;
    }
    double *values = malloc(sizeof(double) * (info.num_nodes > 0 ? info.num_nodes : 1));
    for (int n = 0; n < info.num_nodes; n++) {
      values[n] = value;
    }
    error = goma_set_external_field(ctx, name, GOMA_NODAL, values);
    free(values);
  }
  for (int i = 0; i < mode.num_imports && error == 0; i++) {
    char name[256];
    const char *colon = strchr(mode.imports[i], ':');
    if (colon == NULL || colon - mode.imports[i] >= (long)sizeof(name)) {
      return -1;
    }
    snprintf(name, colon - mode.imports[i] + 1, "%s", mode.imports[i]);
    error = import_field(ctx, name, colon + 1, NULL, 0.0);
    /* a coupled code computes the field from Goma's solution, once Goma has one */
    char field[256];
    double scale;
    if (error == 0 && mode.feedback != NULL && started &&
        sscanf(mode.feedback, "%255[^:]:%255[^:]:%lf", name, field, &scale) == 3 &&
        strncmp(mode.imports[i], name, strlen(name)) == 0 && mode.imports[i][strlen(name)] == ':') {
      error = import_field(ctx, name, colon + 1, field, scale);
    }
  }
  return error;
}

/* Number of owned nodes where the --import-value fields differ from their values
 * before call k */
static int check_import_values(goma_context *ctx, MPI_Comm comm, struct solve_mode mode, int k) {
  int bad = 0;
  for (int i = 0; i < mode.num_import_values; i++) {
    char name[256];
    double value;
    goma_mesh_info info;
    if (import_value(mode, i, k, name, &value) != 0 || goma_mesh_get_info(ctx, &info) != 0) {
      return 1;
    }
    double *values = malloc(sizeof(double) * (info.num_nodes > 0 ? info.num_nodes : 1));
    if (goma_get_post_field(ctx, name, values) != 0) {
      bad++;
    } else {
      for (int n = 0; n < info.num_owned_nodes; n++) {
        bad += values[n] != value;
      }
    }
    free(values);
    if (bad > 0) {
      fprintf(stderr, "goma_lib_sequence: imported field %s is not %g at %d of %d nodes\n", name,
              value, bad, info.num_owned_nodes);
    }
  }
  /* every rank must stop if one does */
  int all = 0;
  MPI_Allreduce(&bad, &all, 1, MPI_INT, MPI_SUM, comm);
  return all;
}

/* Number of nodes where the --import-value fields at the last time step of the output
 * file differ from their values before call k, after the problem is solved */
static int check_import_output(MPI_Comm comm, struct solve_mode mode, int k) {
  int rank, np;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  char path[4096];
  if (np > 1) {
    snprintf(path, sizeof(path), "%s.%d.%d", mode.check_imports, np, rank);
  } else {
    snprintf(path, sizeof(path), "%s", mode.check_imports);
  }
  int bad = 0;
  int cpu_ws = 8, io_ws = 0;
  float version;
  int exoid = ex_open(path, EX_READ, &cpu_ws, &io_ws, &version);
  int num_steps = exoid < 0 ? 0 : ex_inquire_int(exoid, EX_INQ_TIME);
  int num_nodes = exoid < 0 ? 0 : ex_inquire_int(exoid, EX_INQ_NODES);
  int num_vars = 0;
  if (num_steps > 0) {
    ex_get_variable_param(exoid, EX_NODAL, &num_vars);
  }
  double *values = malloc(sizeof(double) * (num_nodes + 1));
  for (int i = 0; i < mode.num_import_values; i++) {
    char name[256];
    double value;
    int var = 0;
    if (import_value(mode, i, k, name, &value) != 0) {
      bad++;
      continue;
    }
    for (int v = 1; v <= num_vars; v++) {
      char var_name[MAX_STR_LENGTH + 1];
      ex_get_variable_name(exoid, EX_NODAL, v, var_name);
      if (strcmp(var_name, name) == 0) {
        var = v;
      }
    }
    if (var == 0) {
      fprintf(stderr, "goma_lib_sequence: rank %d no variable %s in %s\n", rank, name, path);
      bad++;
      continue;
    }
    ex_get_var(exoid, num_steps, EX_NODAL, var, 1, num_nodes, values);
    int differ = 0;
    for (int n = 0; n < num_nodes; n++) {
      differ += values[n] != value;
    }
    if (differ > 0) {
      fprintf(stderr, "goma_lib_sequence: rank %d %s in %s is not %g at %d of %d nodes\n", rank,
              name, path, value, differ, num_nodes);
    }
    bad += differ;
  }
  free(values);
  if (exoid >= 0) {
    ex_close(exoid);
  }
  int all = 0;
  MPI_Allreduce(&bad, &all, 1, MPI_INT, MPI_SUM, comm);
  return all;
}

/* Solve in steps, setting the imported fields again before each call as a coupled
 * code would, returns 0 on success */
static int advance_goma(goma_context *ctx, MPI_Comm comm, struct solve_mode mode) {
  /* the first call starts the problem, a stop time before the start only starts it (so
   * that feedback applies from the first step) */
  int result = mode.advance_steps > 0 && mode.feedback == NULL
                   ? goma_advance_steps(ctx, mode.advance_steps)
                   : goma_advance_to(ctx, -HUGE_VAL);
  double t0, t, dt;
  if (result < 0 && goma_get_time(ctx, &t0, &dt) == 0) {
    fprintf(stderr, "goma_lib_sequence: advance failed\n");
    return result;
  }
  if (result < 0) {
    /* the first call fails without starting for problems that cannot be advanced in steps */
    fprintf(stderr, "goma_lib_sequence: cannot advance, using goma_solve\n");
    return goma_solve(ctx);
  }

  int calls = 1;
  int checked = 0, bad = 0;
  int solved = 0;
  int last = 0; /* the imported values are those set before call last */
  goma_get_time(ctx, &t0, &dt);
  for (int k = 1; result >= 0; k++, calls++) {
    last = k - 1;
    if (mode.check_fields != NULL) {
      int differ = check_fields(ctx, comm, mode.check_fields, &checked);
      bad += differ < 0 ? 1 : differ;
    }
    if (check_import_values(ctx, comm, mode, k - 1) != 0) {
      return -1;
    }
    if (result == 1) {
      break;
    }
    if (set_imports(ctx, mode, k) != 0) {
      return -1;
    }
    if (k == mode.solve_after) {
      /* the values just set must be used for the rest of the problem */
      fprintf(stderr, "goma_lib_sequence: goma_solve after %d calls\n", calls);
      result = goma_solve(ctx);
      solved = 1;
      if (result == 0 && mode.check_imports != NULL && check_import_output(comm, mode, k) != 0) {
        return -1;
      }
      break;
    }
    if (mode.advance_steps > 0) {
      result = goma_advance_steps(ctx, mode.advance_steps);
    } else {
      double target = t0 + k * mode.advance_to;
      result = goma_advance_to(ctx, target);
      if (result == 0 && (goma_get_time(ctx, &t, &dt) != 0 || t != target)) {
        fprintf(stderr, "goma_lib_sequence: goma_advance_to(%.17g) stopped at %.17g\n", target, t);
        return -1;
      }
    }
  }
  fprintf(stderr, "goma_lib_sequence: advanced in %d calls\n", calls);
  if (mode.check_fields != NULL) {
    fprintf(stderr, "goma_lib_sequence: checked %d fields, %d differing values\n", checked, bad);
    if (bad > 0 || checked == 0) {
      return -1;
    }
  }
  if (result < 0 || solved) {
    return result;
  }
  int error = goma_finish(ctx);
  if (error == 0 && mode.check_imports != NULL && check_import_output(comm, mode, last) != 0) {
    return -1;
  }
  return error;
}

static int
run_goma(MPI_Comm comm, int argc, char **argv, enum stop_after stop, struct solve_mode mode) {
  goma_context *ctx = NULL;
  int error = 0;
  for (int i = 0; i < mode.num_declares && error == 0; i++) {
    char name[64];
    const char *colon = strchr(mode.declares[i], ':');
    if (colon == NULL || colon - mode.declares[i] >= (long)sizeof(name)) {
      return -1;
    }
    snprintf(name, colon - mode.declares[i] + 1, "%s", mode.declares[i]);
    error = goma_declare_external_field(name, colon + 1, GOMA_NODAL);
  }
  if (error == 0) {
    error = goma_create(&ctx, comm, argc, argv);
  }
  if (error == 0 && stop >= STOP_AFTER_SETUP) {
    error = goma_setup(ctx);
  }
  if (mode.expect_setup_error) {
    int rejected = ctx != NULL && error < 0;
    if (rejected) {
      rejected = goma_solve(ctx) < 0 && goma_advance_steps(ctx, 1) < 0 && goma_setup(ctx) < 0;
    }
    int destroyed = ctx != NULL ? goma_destroy(&ctx) : -1;
    return rejected && destroyed == 0 ? 0 : -1;
  }
  if (error == 0 && stop >= STOP_AFTER_SETUP) {
    error = set_imports(ctx, mode, 0);
  }
  if (error == 0 && stop >= STOP_AFTER_SOLVE) {
    if (mode.advance_steps > 0 || mode.advance_to > 0) {
      error = advance_goma(ctx, comm, mode);
    } else {
      error = goma_solve(ctx);
    }
  }
  if (ctx != NULL) {
    goma_destroy(&ctx);
  }
  return error;
}

int main(int argc, char **argv) {
  int status = 0;
  MPI_Init(&argc, &argv);

  char start_dir[4096];
  if (getcwd(start_dir, sizeof(start_dir)) == NULL || argc < 2) {
    fprintf(stderr,
            "usage: %s [--ranks=<n>] <dir> [--stop-after=create|setup] "
            "[--advance-steps=<n> | --advance-to=<dt>] [--solve-after=<n>] [--log=<file>] "
            "<goma args> "
            "[-- <dir> ...] ...\n",
            argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int run = 0;
  int first = 1;
  MPI_Comm comm = MPI_COMM_WORLD;
  if (strncmp(argv[1], "--ranks=", 8) == 0) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int color = world_rank < atoi(argv[1] + 8) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &comm);
    first = 2;
  }
  if (comm == MPI_COMM_NULL) {
    /* not one of the ranks running Goma */
    MPI_Finalize();
    return 0;
  }

  int rank;
  MPI_Comm_rank(comm, &rank);
  goma_library_initialize(comm);

  while (first < argc) {
    int last = first;
    while (last < argc && strcmp(argv[last], "--") != 0) {
      last++;
    }
    const char *dir = argv[first];
    enum stop_after stop = STOP_AFTER_SOLVE;
    const char *log = NULL;
    struct solve_mode mode = {0};
    const char *imports[MAX_IMPORTS];
    int num_imports = 0;
    const char *declares[MAX_IMPORTS];
    int num_declares = 0;
    const char *import_values[MAX_IMPORTS];
    int num_import_values = 0;
    int expect_error = 0;
    int run_first = 0;
    int run_size = 0; /* 0: all of comm */
    while (first + 1 < last && strncmp(argv[first + 1], "--", 2) == 0) {
      const char *opt = argv[first + 1];
      if (strncmp(opt, "--stop-after=", 13) == 0) {
        const char *stage = opt + 13;
        if (strcmp(stage, "create") == 0) {
          stop = STOP_AFTER_CREATE;
        } else if (strcmp(stage, "setup") == 0) {
          stop = STOP_AFTER_SETUP;
        } else {
          fprintf(stderr, "unknown stage %s\n", stage);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      } else if (strncmp(opt, "--advance-steps=", 16) == 0) {
        mode.advance_steps = atoi(opt + 16);
      } else if (strncmp(opt, "--advance-to=", 13) == 0) {
        mode.advance_to = atof(opt + 13);
      } else if (strncmp(opt, "--import-value=", 15) == 0 && num_import_values < MAX_IMPORTS) {
        import_values[num_import_values++] = opt + 15;
      } else if (strncmp(opt, "--import-value-step=", 20) == 0) {
        mode.import_value_step = atof(opt + 20);
      } else if (strncmp(opt, "--solve-after=", 14) == 0) {
        mode.solve_after = atoi(opt + 14);
      } else if (strncmp(opt, "--check-imports=", 16) == 0) {
        mode.check_imports = opt + 16;
      } else if (strncmp(opt, "--import=", 9) == 0 && num_imports < MAX_IMPORTS) {
        imports[num_imports++] = opt + 9;
      } else if (strncmp(opt, "--declare=", 10) == 0 && num_declares < MAX_IMPORTS) {
        declares[num_declares++] = opt + 10;
      } else if (strncmp(opt, "--feedback=", 11) == 0) {
        mode.feedback = opt + 11;
      } else if (strncmp(opt, "--check-fields=", 15) == 0) {
        mode.check_fields = opt + 15;
      } else if (strcmp(opt, "--expect-setup-error") == 0) {
        mode.expect_setup_error = 1;
      } else if (strcmp(opt, "--expect-error") == 0) {
        expect_error = 1;
      } else if (strncmp(opt, "--run-ranks=", 12) == 0) {
        if (sscanf(opt + 12, "%d:%d", &run_first, &run_size) != 2) {
          fprintf(stderr, "bad option %s\n", opt);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      } else if (strncmp(opt, "--log=", 6) == 0) {
        log = opt + 6;
      } else {
        fprintf(stderr, "unknown option %s\n", opt);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      first++;
    }
    mode.num_imports = num_imports;
    mode.imports = imports;
    mode.num_import_values = num_import_values;
    mode.import_values = import_values;
    mode.num_declares = num_declares;
    mode.declares = declares;
    /* goma expects argv[0] to be the program name */
    char *run_argv[last - first + 1];
    run_argv[0] = argv[0];
    for (int i = first + 1; i < last; i++) {
      run_argv[i - first] = argv[i];
    }
    run_argv[last - first] = NULL;

    MPI_Comm run_comm = comm;
    if (run_size > 0) {
      int in_run = rank >= run_first && rank < run_first + run_size;
      MPI_Comm_split(comm, in_run ? 0 : MPI_UNDEFINED, rank, &run_comm);
      if (run_comm == MPI_COMM_NULL) {
        fprintf(stderr, "goma_lib_sequence: run %d in %s skipped by this rank\n", run, dir);
        run++;
        first = last + 1;
        continue;
      }
    }
    int run_rank;
    MPI_Comm_rank(run_comm, &run_rank);

    int saved_stdout = -1;
    if (log != NULL && run_rank == 0) {
      int fd = open(log, O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (fd < 0) {
        fprintf(stderr, "could not open %s\n", log);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      fflush(stdout);
      saved_stdout = dup(STDOUT_FILENO);
      dup2(fd, STDOUT_FILENO);
      close(fd);
    }
    if (chdir(dir) != 0) {
      fprintf(stderr, "could not change to directory %s\n", dir);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /*
     * GOMA_LIB_SEQUENCE_TRACE_RUN=<n> asks a preloaded allocation tracer that
     * defines leaktrace_enable(int) to record run n, allocations still live
     * when it is disabled were leaked by that run
     */
    void (*leaktrace_enable)(int) = (void (*)(int))dlsym(RTLD_DEFAULT, "leaktrace_enable");
    const char *trace_run = getenv("GOMA_LIB_SEQUENCE_TRACE_RUN");
    int trace = leaktrace_enable != NULL && trace_run != NULL && atoi(trace_run) == run;
    if (trace)
      leaktrace_enable(1);
    int error = run_goma(run_comm, last - first, run_argv, stop, mode);
    if (trace)
      leaktrace_enable(0);
    if (saved_stdout >= 0) {
      fflush(stdout);
      dup2(saved_stdout, STDOUT_FILENO);
      close(saved_stdout);
    }
    fprintf(stderr, "goma_lib_sequence: run %d in %s returned %d\n", run, dir, error);
    if (expect_error) {
      if (error >= 0) {
        fprintf(stderr, "goma_lib_sequence: run %d in %s did not return an error\n", run, dir);
      }
      error = error < 0 ? 0 : 1;
    }
#ifdef __GLIBC__
    /* heap still in use after goma_destroy, growth between runs is leaked memory */
#if __GLIBC_PREREQ(2, 33)
    struct mallinfo2 mi = mallinfo2();
    size_t heap_in_use = mi.uordblks;
#else
    /* mallinfo2 needs glibc 2.33, mallinfo fields are int and wrap past 2 GiB */
    struct mallinfo mi = mallinfo();
    size_t heap_in_use = (size_t)(unsigned int)mi.uordblks;
#endif
    fprintf(stderr, "goma_lib_sequence: run %d heap in use %zu bytes\n", run, heap_in_use);
#endif
    if (error != 0) {
      status = 1;
    }
    if (chdir(start_dir) != 0) {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (run_comm != comm) {
      MPI_Comm_free(&run_comm);
    }
    run++;
    first = last + 1;
  }

  goma_library_finalize();
  if (comm != MPI_COMM_WORLD) {
    MPI_Comm_free(&comm);
  }
  MPI_Finalize();
  return status;
}
