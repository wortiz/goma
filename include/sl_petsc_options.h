#ifndef GOMA_SL_PETSC_OPTIONS_H
#define GOMA_SL_PETSC_OPTIONS_H

#ifdef GOMA_ENABLE_PETSC
#include <petscsys.h>
#ifdef I
#undef I
#endif

/*
 * Initialize PETSc unless it is initialized already (by an earlier run or
 * the host), on a duplicate of goma_comm.  goma_petsc_finalize() finalizes
 * PETSc only if Goma initialized it.
 */
PetscErrorCode goma_petsc_initialize(PetscBool *initialized_by_run);
void goma_petsc_finalize(void);

/*
 * PETSc options of one Goma run: the problem directory's .petscrc and
 * petscrc files and the -petsc command line option.  The options database
 * outlives the run when PETSc stays initialized, so the options a run
 * inserted are recorded and goma_petsc_restore_options() restores their
 * previous values.
 */

/* initialized_by_run: PETSc was initialized by this run, so it has already
 * read the petscrc files in the current directory */
PetscErrorCode goma_petsc_insert_run_options(PetscBool initialized_by_run);
PetscErrorCode goma_petsc_restore_options(void);
#endif

#endif
