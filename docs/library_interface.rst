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
*Goma* input. Within a run, the host can advance the problem in steps and
exchange fields with it, to couple *Goma* to another code (see `Solving in
steps`_ and `Mesh and fields`_).

Building
########

The interface is part of the libraries built with *Goma*. ``cmake --install``
installs them, ``goma.h`` (in ``include/goma``) and a CMake package, so a host
program built with CMake can use an installed *Goma*:

.. code-block:: cmake

   find_package(Goma REQUIRED)
   target_link_libraries(host PRIVATE Goma::goma)

with the install prefix in ``CMAKE_PREFIX_PATH``. ``Goma::goma`` brings in
*Goma*'s libraries, the third party libraries it was configured with (the
package finds Trilinos, SEACAS, Omega_h and PETSc where *Goma*'s build found
them) and MPI. The other libraries *Goma* links (ARPACK, METIS, MUMPS,
UMFPACK, ...) are recorded in the package by their absolute paths, so the
install is not relocatable: build the host on the machine *Goma* was built on,
with those libraries still where *Goma*'s build found them. Inside *Goma*'s own build, link the ``goma_library`` target, as
``tests/library/goma_lib_sequence.c`` does in ``tests/CMakeLists.txt``.

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
   function. It is collective over ``comm``, which must hold every process
   that will run a problem (see `MPI and PETSc`_).

``goma_create``
   Parses a *Goma* command line and reads the input deck. ``argv[0]`` is the
   program name and the remaining arguments are the usual command-line options,
   for example ``-i input -a``. Relative file names in the deck are opened from
   the current working directory, so change to the problem directory first.

``goma_setup``
   Reads the mesh, decomposes it when running in parallel and sets up the
   problem.

``goma_solve``
   Solves the problem and writes the output files. It returns an error when
   the solver gives up: a steady solve that does not converge, a time step
   that falls below the deck's ``Minimum time step`` or a failed element
   quality check. The output written until then is kept.

``goma_destroy``
   Frees the context and the memory allocated for the problem, and restores
   *Goma*'s global variables to their initial values so the next
   ``goma_create`` starts from the same state as a new process. Sets
   ``*ctx`` to ``NULL``. It can be called after any stage, so a host can
   stop after ``goma_create`` or ``goma_setup``, or after one of them
   returned an error.

``goma_library_finalize``
   Finalizes third-party libraries that *Goma* initialized. Call it once on
   every process of the ``goma_library_initialize`` communicator, after the
   last ``goma_destroy`` and before ``MPI_Finalize``.

All functions return 0 on success and a negative value on failure.

*Goma* keeps its state in global variables, so only one context can exist at a
time: ``goma_create`` returns an error while another context is alive. Running
problems concurrently in one process (for example from several threads) is not
supported.

Solving in steps
################

::

   int goma_advance_steps(goma_context *ctx, int num_steps);
   int goma_advance_to(goma_context *ctx, double time);
   int goma_get_time(goma_context *ctx, double *time, double *delta_t);
   int goma_finish(goma_context *ctx);

Instead of ``goma_solve``, a host can advance a problem a few time steps at a
time, for example to run *Goma* alongside another code. Output is written at
the deck's print frequency as usual. A problem advanced with
``goma_advance_steps`` gives the same results as ``goma_solve``.

``goma_advance_steps``
   Takes ``num_steps`` (at least 1) successful time steps.

``goma_advance_to``
   Takes time steps until the simulation time reaches ``time``, shortening the
   last step to land on it exactly. The following step goes back to the size
   the time step control chose before it was shortened. A ``time`` at or
   before the current time takes no step.

``goma_get_time``
   The simulation time of the last converged step and the size of the next
   time step.

``goma_finish``
   Frees the solver and finishes the output files. ``goma_destroy`` calls it
   when the host did not. Calling it before the problem has reached its end
   stops the problem at the current time.

Both advance functions return 0 when the problem can be advanced further and
1 when it has reached its end (the maximum time or number of time steps, or a
steady state); further calls then return 1 without doing anything. They
return an error when the solver gives up, as ``goma_solve`` does, and so do
further calls; ``goma_finish`` still finishes the output files. A steady
problem is solved by the first call. The time step size carries over from one
call to the next, and ``goma_solve`` called after them solves the rest of the
problem.

Steady or transient problems with a single matrix and no continuation,
hunting or LOCA, and problems solved by the segregated solver (several
matrices), can be advanced in steps. For other problems the first call returns
an error and the problem can still be solved with ``goma_solve``. A steady
segregated problem is solved by the first call, like a steady fully coupled
one.

A host loop that advances a transient problem every 0.1 time units:

.. code-block:: c

   double t, dt;
   int status = goma_advance_to(ctx, -HUGE_VAL); /* starts the problem */
   goma_get_time(ctx, &t, &dt);
   while (status == 0) {
     t += 0.1;
     status = goma_advance_to(ctx, t);
     /* ... work with the other code at time t ... */
   }
   goma_finish(ctx);

Mesh and fields
###############

::

   int goma_mesh_get_info(goma_context *ctx, goma_mesh_info *info);
   int goma_mesh_get_coordinates(goma_context *ctx, double *xyz);
   int goma_mesh_get_node_global_ids(goma_context *ctx, long long *ids);
   int goma_mesh_get_elem_global_ids(goma_context *ctx, long long *ids);
   int goma_mesh_get_elem_owners(goma_context *ctx, int *owners);
   int goma_mesh_get_block(goma_context *ctx, int block, int *block_id, int *num_elems,
                           int *nodes_per_elem, const int **connectivity);
   int goma_get_field(goma_context *ctx, const char *name, int component, double *values);
   int goma_get_post_field(goma_context *ctx, const char *name, double *values);
   int goma_set_external_field(goma_context *ctx, const char *name,
                               goma_field_location location, const double *values);

A host that couples *Goma* to another code exchanges values at *Goma*'s nodes
and elements; mapping them to and from the other code's mesh is the host's
job. The mesh functions describe the part of the mesh on the calling process,
after ``goma_setup``:

* Nodes are numbered locally from 0. The nodes the process owns come first
  (``num_owned_nodes`` of them), then ghost nodes owned by other processes.
* Elements are numbered locally from 0, element block by element block. In
  parallel, elements along a processor boundary are on both processes;
  ``goma_mesh_get_elem_owners`` gives the process that owns each one.
* Global ids are the 1-based node and element numbers of the undecomposed
  mesh, and match the node and element maps of *Goma*'s parallel output files
  (which number their nodes differently).
* Coordinates are those of the mesh as read, without mesh displacements, which
  are solution variables (``MESH_DISPLACEMENT1`` ...).

A problem with mesh adaptation changes the mesh while it advances.

After ``goma_advance_steps`` or ``goma_advance_to``, ``goma_get_field``
returns a solution variable at every local node as it would be written to the
output file. The name is a variable name (``VELOCITY1`` or ``VX``; for
``MASS_FRACTION`` or ``Y`` the species number is ``component``) or a nodal
variable name of the output file. ``goma_get_post_field`` returns a nodal post
processing variable that the deck asks for, named as in the output file; it
computes the post processing variables and must be called on all processes.
The values are those of the last converged step: a level set renormalization
or a stability analysis that runs after the output of a step can make them
differ from the output file.

``goma_set_external_field`` sets an external field that the deck declares
with ``IMPORT`` or ``IMPORT_EV`` in place of a file name (see
:doc:`/problem_description_file/general_specifications/external_field`)::

   External Field = SOURCE Q1 IMPORT

``GOMA_NODAL`` values are given at every local node, but only the owned
nodes' values are used; *Goma* copies them to the ghost nodes. ``GOMA_ELEMENT``
values are given at every local element (owned and ghost) and averaged to the
nodes, each element weighted by its size (the product of its widths between
face centroids). Call it after ``goma_setup`` and again whenever the values change;
*Goma* uses them from the next ``goma_advance_*`` call on. ``goma_solve`` and
``goma_advance_*`` return an error while an imported field has no values.
When mesh adaptation changes the mesh during a ``goma_advance_*`` call, the
values set before it were for the old mesh: the field keeps its values as
adaptation transferred them to the new mesh until the host sets it again, with
values for the new mesh (see ``goma_mesh_get_info``).

A host can also declare an imported field itself, so the deck needs no card
for it. ``goma_declare_external_field`` is called before ``goma_create``, with
the same arguments on every process::

   goma_declare_external_field("SOURCE", "Q1", GOMA_NODAL);
   goma_create(&ctx, comm, argc, argv);

This is the same as adding the card ``External Field = SOURCE Q1 IMPORT``
(``IMPORT_EV`` for ``GOMA_ELEMENT``) after the deck's ``External Field``
cards. The field exists while the deck is read, so material property cards can
use it, for example ``Conductivity = EXTERNAL_FIELD SOURCE 1.0``. A declaration
applies to the next problem only: ``goma_destroy`` forgets it. It returns an
error for a name longer than 20 characters or declared twice, an unknown
interpolation (``TABLE`` and ``N1`` are not allowed) or too many external
fields. A field that the deck also declares is a fatal error when the deck is
read.

A coupling loop that sends *Goma*'s temperature to another code and gets a
source term back every 0.1 time units:

.. code-block:: c

   goma_mesh_info mesh;
   goma_mesh_get_info(ctx, &mesh);
   double *temperature = malloc(mesh.num_nodes * sizeof(double));
   double *source = malloc(mesh.num_nodes * sizeof(double));

   other_code_source(source);           /* host maps the other code's values to Goma's nodes */
   goma_set_external_field(ctx, "SOURCE", GOMA_NODAL, source);
   double t, dt;
   int status = goma_advance_to(ctx, -HUGE_VAL);
   goma_get_time(ctx, &t, &dt);
   while (status == 0) {
     t += 0.1;
     status = goma_advance_to(ctx, t);
     goma_get_field(ctx, "TEMPERATURE", 0, temperature);
     other_code_advance(t, temperature, source);
     goma_set_external_field(ctx, "SOURCE", GOMA_NODAL, source);
   }
   goma_finish(ctx);

MPI and PETSc
#############

The host program owns MPI. It calls ``MPI_Init`` and ``MPI_Finalize``; *Goma*
never does in library mode, including on error.

Each run communicates only over its own duplicate of the communicator passed
to ``goma_create``, which can hold any subset of the processes, for example one
made with ``MPI_Comm_split``. The other processes are free to do other work,
including running their own *Goma* problem on another communicator. File names
of a parallel run are numbered by rank in that communicator.

If PETSc is already initialized when ``goma_library_initialize`` is called,
the host owns it and *Goma* leaves it initialized; *Goma*'s PETSc objects then
live on the run's communicator, which must contain only processes of
``PETSC_COMM_WORLD``. Otherwise ``goma_library_initialize`` initializes PETSc
with ``PETSC_COMM_WORLD`` set to a duplicate of its communicator, so runs on
any subset of those processes can use it, and ``goma_library_finalize`` calls
``PetscFinalize``. A host that uses PETSc itself must initialize it before
``goma_library_initialize``. PETSc reads ``PETSC_OPTIONS``, ``~/.petscrc`` and
the ``.petscrc`` and ``petscrc`` files of the current directory when it is
initialized, as ``PetscInitialize`` always does.

Each run reads the ``.petscrc`` and ``petscrc`` files in its directory and the
``-petsc`` command-line option, as a standalone ``goma`` run does.
``goma_destroy`` restores the options that the run set to their previous
values, so options of one problem do not carry over to the next, and options
set by the host or read when PETSc was initialized are kept.

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

``--advance-steps N`` solves every problem with ``goma_advance_steps(N)``
instead of ``goma_solve``, and ``--advance-to DT`` with ``goma_advance_to`` at
every ``DT`` of simulation time; problems that cannot be advanced in steps are
solved with ``goma_solve`` and listed at the end. ``--advance-to`` changes the
time steps, so its output only matches the blessed results when every ``DT``
falls on a time step.

``--check-fields`` (with ``--advance-steps`` or ``--advance-to``) compares
``goma_get_field`` and ``goma_get_post_field`` after every call with each
nodal variable of the output file when its last step is the current time.
``--import-fields`` replaces the file of every ``External Field`` card that
reads a fixed field from a mesh file with ``IMPORT``; the host reads the field
from the file and sets it with ``goma_set_external_field`` instead; with
``--declare-fields`` it also removes the card and declares the field with
``goma_declare_external_field``. With
``--feedback NAME:FIELD:SCALE`` the host adds ``SCALE`` times *Goma*'s
``FIELD`` to the imported field ``NAME`` before every advance, a coupling loop
whose results differ from the blessed ones; ``--compare-runs`` then checks that
the runs of a problem give exactly the same results instead.

``--extra-ranks 1`` starts one more MPI process than each problem needs and
runs *Goma* on a communicator without it, to check that *Goma* does not
communicate outside the communicator it is given. With ``--shift-ranks`` the
library is initialized on all the processes instead and each run moves one
process further, so later runs use processes that earlier ones did not.

``--declare-value NAME:VALUE[:INTERP]`` declares an imported nodal field
(interpolation ``INTERP``, ``Q1`` by default) and sets it to ``VALUE`` once;
with ``--advance-steps`` it must still be ``VALUE`` after every call, also
after mesh adaptation. The output then has the extra field, so it is not
compared. ``--import-value-step STEP`` instead sets the field again before
every call, to ``VALUE + k * STEP`` before call ``k``, sized for the mesh as
earlier calls adapted it, and checks those values after the call;
``--check-imports`` checks the field at the last step of the output file
against the values set last. ``--solve-after N`` finishes each problem with
``goma_solve`` after ``N`` advance calls, setting the imported fields first.

``--set-card "NAME=VALUE"`` sets a card of every problem's input deck, and
``--expect-failure`` then requires the solve to return an error instead of
comparing the output, for cards that make the solver give up, for example
``--set-card "Minimum time step=1.0e6"``.

With ``-DENABLE_TESTING=ON`` and the ``guts`` submodule checked out, ctest runs
a few of these problems (label ``library``), and ``goma_lib_install`` installs
*Goma* and builds and runs ``tests/library/install``, a host program using
``find_package(Goma)``::

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

The ``LIBRARY_MODE`` preprocessor macro of earlier versions, an interface for
coupling *Goma* to the ANIMAS code that called ``solve_problem`` repeatedly
for the same problem, has been removed; this interface replaces it. Its
``Export Field`` card and the ``exp`` option of the post processing cards are
ignored with a warning (use ``goma_get_field`` and ``goma_get_post_field``),
and the update of an imported porosity field with the porous liquid pressure
is no longer available.
