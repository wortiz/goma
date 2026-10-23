#ifndef GOMA_SL_PETSC_OPTIONS_H
#define GOMA_SL_PETSC_OPTIONS_H

#ifdef GOMA_ENABLE_PETSC
#include <petscsys.h>
#ifdef I
#undef I
#endif

/*
 * Initialize PETSc unless it is initialized already (by the host), on a
 * duplicate of comm, collective over comm.  Called by
 * goma_library_initialize(), so PETSc is initialized on every process that
 * may run a problem.  goma_petsc_finalize() finalizes PETSc only if Goma
 * initialized it.
 */
PetscErrorCode goma_petsc_initialize(MPI_Comm comm);
void goma_petsc_finalize(void);

/*
 * PETSc options of one Goma run: the problem directory's .petscrc and
 * petscrc files and the -petsc command line option.  The options database
 * outlives the run, so the options a run inserted are recorded and
 * goma_petsc_restore_options() restores their previous values.
 */
PetscErrorCode goma_petsc_insert_run_options(void);
PetscErrorCode goma_petsc_restore_options(void);
#endif

#endif
