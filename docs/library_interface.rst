=================
Library Interface
=================

*Goma* can be built into another program and called through a C interface
declared in ``goma.h``. The host program can run several problems one after
another in the same process, for example a parameter study or a workflow that
runs a different input deck at each step. Each run is independent: *Goma*
reads its input deck and mesh, solves, writes its output files and then frees
everything before the next run starts. Problems do not exchange data in
memory; a later run can read an earlier run's output files like any other
*Goma* input.

Building
########

The interface is part of the ``goma`` library built with *Goma*. Link the host
program against ``goma``, ``goma_user`` and ``goma_util`` and the third party
libraries *Goma* was configured with, as ``tests/library/goma_lib_sequence.c``
does in ``tests/CMakeLists.txt``. ``goma.h`` is installed with the other
headers.

Lifecycle
#########

::

   int goma_library_initialize(MPI_Comm comm);
   int goma_create(goma_context **ctx, MPI_Comm comm, int argc, char **argv);
   int goma_setup(goma_context *ctx);
   int goma_solve(goma_context *ctx);
   int goma_destroy(goma_context **ctx);
   int goma_library_finalize(void);

``goma_library_initialize``
   Call once per process, after ``MPI_Init`` and before any other ``goma_``
   function.

``goma_create``
   Parses a *Goma* command line and reads the input deck. ``argv[0]`` is the
   program name and the remaining arguments are the usual command-line options,
   for example ``-i input -a``. Relative file names in the deck are opened from
   the current working directory, so change to the problem directory first.

``goma_setup``
   Reads the mesh, decomposes it when running in parallel and sets up the
   problem.

``goma_solve``
   Solves the problem and writes the output files.

``goma_destroy``
   Frees the context and the memory allocated for the problem, and restores
   *Goma*'s global variables to their initial values so the next
   ``goma_create`` starts from the same state as a new process. Sets
   ``*ctx`` to ``NULL``. It can be called after any stage, so a host can
   stop after ``goma_create`` or ``goma_setup``, or after one of them
   returned an error.

``goma_library_finalize``
   Finalizes third-party libraries that *Goma* initialized. Call it once,
   after the last ``goma_destroy`` and before ``MPI_Finalize``.

All functions return 0 on success and a negative value on failure.

*Goma* keeps its state in global variables, so only one context can exist at a
time: ``goma_create`` returns an error while another context is alive. Running
problems concurrently in one process (for example from several threads) is not
supported.

Running more than one problem
   Until the library interface is declared stable, a second ``goma_create`` in
   the same process returns an error unless the environment variable
   ``GOMA_EXPERIMENTAL_MULTIPLE_RUNS`` is set.

MPI and PETSc
#############

The host program owns MPI. It calls ``MPI_Init`` and ``MPI_Finalize``; *Goma*
never does in library mode, including on error.

Each run communicates only over its own duplicate of the communicator passed
to ``goma_create``, which can hold any subset of the processes, for example one
made with ``MPI_Comm_split``. The other processes are free to do other work,
including running their own *Goma* problem on another communicator. File names
of a parallel run are numbered by rank in that communicator.

If PETSc is already initialized when a problem first uses it, the host owns
it and *Goma* leaves it initialized; *Goma*'s PETSc objects then live on the
run's communicator, which must contain only processes of ``PETSC_COMM_WORLD``.
Otherwise *Goma* initializes PETSc with ``PETSC_COMM_WORLD`` set to a duplicate
of the communicator of that run, and ``goma_library_finalize`` calls
``PetscFinalize``. Later runs that use PETSc must then use the same processes
or a subset of them.

Each run reads the ``.petscrc`` and ``petscrc`` files in its directory and the
``-petsc`` command-line option, as a standalone ``goma`` run does, even when
PETSc was initialized by an earlier run or by the host. ``goma_destroy``
restores the options that the run set to their previous values, so options of
one problem do not carry over to the next, and options set by the host,
``PETSC_OPTIONS`` or ``~/.petscrc`` are kept. One exception: in the run that
initializes PETSc, an option in the problem's ``petscrc`` that also appears in
``PETSC_OPTIONS`` or ``~/.petscrc`` is removed when the run ends.

Errors
######

Fatal errors still end the process: *Goma* prints its error message and calls
``MPI_Abort`` (or ``exit`` in a serial build), as the ``goma`` executable
does. A host program that must survive a failed run should start *Goma* in a
separate process. Floating point exception traps that *Goma* enables are
restored to the host's settings by ``goma_destroy``.

Example
#######

A host program that runs two problems, each in its own directory:

.. code-block:: c

   #include <mpi.h>
   #include <stdio.h>
   #include <unistd.h>

   #include "goma.h"

   static int run_goma(const char *dir, int argc, char **argv) {
     char cwd[4096];
     if (getcwd(cwd, sizeof(cwd)) == NULL || chdir(dir) != 0) {
       return -1;
     }
     goma_context *ctx = NULL;
     int error = goma_create(&ctx, MPI_COMM_WORLD, argc, argv);
     if (error == 0) {
       error = goma_setup(ctx);
     }
     if (error == 0) {
       error = goma_solve(ctx);
     }
     if (ctx != NULL) {
       goma_destroy(&ctx);
     }
     if (chdir(cwd) != 0) {
       return -1;
     }
     return error;
   }

   int main(int argc, char **argv) {
     MPI_Init(&argc, &argv);
     goma_library_initialize(MPI_COMM_WORLD);

     char *flow[] = {"host", "-i", "input", NULL};
     char *heat[] = {"host", "-i", "heat.inp", "-a", NULL};
     int error = run_goma("flow", 3, flow);
     if (error == 0) {
       error = run_goma("heat", 4, heat);
     }

     goma_library_finalize();
     MPI_Finalize();
     return error == 0 ? 0 : 1;
   }

``tests/library/goma_lib_sequence.c`` is a complete host program that runs any
list of problems given on its command line.

Testing
#######

``scripts/goma_lib_regression.py`` runs the GUTS problems through
``goma_lib_sequence`` and compares the output of every run with the blessed
results the same way GUTS does. By default each problem is run twice in one
process; ``--mode chain`` runs shuffled chains of different problems in one
process instead. When a problem's blessed standard output lists PETSc options,
the run must list the same ones::

   scripts/goma_lib_regression.py --driver build/tests/goma_lib_sequence
   scripts/goma_lib_regression.py --driver build/tests/goma_lib_sequence --mode chain

``--extra-ranks 1`` starts one more MPI process than each problem needs and
runs *Goma* on a communicator without it, to check that *Goma* does not
communicate outside the communicator it is given.

With ``-DENABLE_TESTING=ON`` and the ``guts`` submodule checked out, ctest runs
a few of these problems (label ``library``)::

   ctest --test-dir build -L library

Configuring with ``-DGOMA_CHECK_GLOBAL_RESET=ON`` makes ``goma_destroy`` compare
*Goma*'s writable global data with its initial values and write any
differences to ``goma_reset_check.<rank>.txt``;
``scripts/resolve_reset_check.py`` names the variables and
``goma_lib_regression.py`` fails a run that left any behind. A global variable
added to *Goma* must be restored in its source file's ``*_free_globals``
function (see ``include/goma_reset.h``) for these tests to pass.

``LIBRARY_MODE``
################

The ``LIBRARY_MODE`` preprocessor macro found in some source files is an older,
unrelated interface for coupling *Goma* to another code, which calls
``solve_problem`` repeatedly for the same problem. It is not used by the
interface described here.
