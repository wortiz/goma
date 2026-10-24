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

#ifdef GOMA_ENABLE_PETSC
#include <stdlib.h>
#include <string.h>

#include "rf_io.h"
#include "rf_mp.h"
#include "sl_petsc_options.h"

/* Communicator PETSc was initialized on when Goma initialized it */
static MPI_Comm goma_petsc_comm = MPI_COMM_NULL;

PetscErrorCode goma_petsc_initialize(MPI_Comm comm) {
  PetscErrorCode err;
  PetscBool initialized = PETSC_FALSE;
  err = PetscInitialized(&initialized);
  CHKERRQ(err);
  if (!initialized) {
    /* PETSc outlives the runs, so it gets its own communicator, duplicated on all of comm's
     * processes so that runs on any subset of them can use it */
    MPI_Comm_dup(comm, &goma_petsc_comm);
    PETSC_COMM_WORLD = goma_petsc_comm;
    err = PetscInitializeNoArguments();
    CHKERRQ(err);
  }
  return 0;
}

void goma_petsc_finalize(void) {
  if (goma_petsc_comm != MPI_COMM_NULL) {
    PetscFinalize();
    MPI_Comm_free(&goma_petsc_comm);
  }
}

/* An option set by the current run and its value before the run */
typedef struct {
  char *name;
  char *value; /* NULL for an option without a value */
  PetscBool was_set;
} saved_option;

static saved_option *saved_options = NULL;
static int num_saved_options = 0;

static char *copy_string(const char *s) {
  if (s == NULL) {
    return NULL;
  }
  char *copy = malloc(strlen(s) + 1);
  strcpy(copy, s);
  return copy;
}

/* Record the current value of every option named in opts that is not recorded yet */
static PetscErrorCode save_options(const char *opts) {
  PetscErrorCode err;
  char *copy = copy_string(opts);
  for (char *token = strtok(copy, " \t\r\n"); token != NULL; token = strtok(NULL, " \t\r\n")) {
    PetscBool valid = PETSC_FALSE;
    err = PetscOptionsValidKey(token, &valid);
    CHKERRQ(err);
    if (!valid) {
      continue;
    }
    PetscBool saved = PETSC_FALSE;
    for (int i = 0; i < num_saved_options && !saved; i++) {
      saved = strcmp(saved_options[i].name, token) == 0;
    }
    if (saved) {
      continue;
    }
    const char *value = NULL;
    PetscBool set = PETSC_FALSE;
    err = PetscOptionsFindPair(NULL, NULL, token, &value, &set);
    CHKERRQ(err);
    saved_options = realloc(saved_options, (num_saved_options + 1) * sizeof(saved_option));
    saved_options[num_saved_options].name = copy_string(token);
    saved_options[num_saved_options].value = set ? copy_string(value) : NULL;
    saved_options[num_saved_options].was_set = set;
    num_saved_options++;
  }
  free(copy);
  return 0;
}

/* Record and insert the options of a petscrc file in the current directory */
static PetscErrorCode insert_options_file(const char *file) {
  PetscErrorCode err;
  PetscOptions file_options;
  char *opts;
  err = PetscOptionsCreate(&file_options);
  CHKERRQ(err);
  err = PetscOptionsInsertFile(goma_comm, file_options, file, PETSC_FALSE);
  CHKERRQ(err);
  err = PetscOptionsGetAll(file_options, &opts);
  CHKERRQ(err);
  err = PetscOptionsDestroy(&file_options);
  CHKERRQ(err);
  err = save_options(opts);
  CHKERRQ(err);
  err = PetscFree(opts);
  CHKERRQ(err);
  err = PetscOptionsInsertFile(goma_comm, NULL, file, PETSC_FALSE);
  CHKERRQ(err);
  return 0;
}

PetscErrorCode goma_petsc_insert_run_options(void) {
  PetscErrorCode err;
  /* the files PetscInitialize reads from the current directory */
  PetscBool skip_petscrc = PETSC_FALSE;
  err = PetscOptionsGetBool(NULL, NULL, "-skip_petscrc", &skip_petscrc, NULL);
  CHKERRQ(err);
  if (!skip_petscrc) {
    err = insert_options_file(".petscrc");
    CHKERRQ(err);
    err = insert_options_file("petscrc");
    CHKERRQ(err);
  }
  if (GomaPetscOptions != NULL) {
    err = save_options(GomaPetscOptions);
    CHKERRQ(err);
    err = PetscOptionsInsertString(NULL, GomaPetscOptions);
    CHKERRQ(err);
  }
  return 0;
}

PetscErrorCode goma_petsc_restore_options(void) {
  PetscErrorCode err;
  PetscBool petsc_initialized = PETSC_FALSE;
  err = PetscInitialized(&petsc_initialized);
  CHKERRQ(err);
  for (int i = num_saved_options - 1; i >= 0; i--) {
    if (petsc_initialized) {
      if (saved_options[i].was_set) {
        err = PetscOptionsSetValue(NULL, saved_options[i].name, saved_options[i].value);
        CHKERRQ(err);
      } else {
        err = PetscOptionsClearValue(NULL, saved_options[i].name);
        CHKERRQ(err);
      }
    }
    free(saved_options[i].name);
    free(saved_options[i].value);
  }
  free(saved_options);
  saved_options = NULL;
  num_saved_options = 0;
  return 0;
}
#endif
