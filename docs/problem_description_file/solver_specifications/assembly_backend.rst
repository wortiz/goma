Assembly Backend
================

::

   Assembly Backend = {legacy | kokkos}

Select the assembly implementation for the complete coupled system. The default
is ``legacy``, which preserves the existing assembly behavior and works with
``Use AutoDiff Assembly`` as before.

``kokkos`` reserves the experimental device assembly path. Build its
infrastructure with ``-DENABLE_KOKKOS_ASSEMBLY=ON``; this requires C++20 and a
Trilinos installation containing Kokkos, Tpetra, and Stratimikos, with the latter
two enabled in Goma. Legacy-only builds continue to use C++17 and C11.

The Kokkos assembly path is still under development. A build without the
feature rejects ``kokkos`` during input parsing. An enabled build accepts the
selection and runs collective preflight after problem setup, then rejects
execution before assembly. It never falls back to legacy assembly. Kokkos
with ``Use AutoDiff Assembly = yes`` is also unavailable; that incompatibility
is reported with the other preflight diagnostics.

Enabled builds initialize Kokkos for the Goma library lifetime, borrowing an
already initialized host runtime when present. Assembly contexts own device
error storage and must be destroyed before finalization.

Flow expressions
----------------

The internal expression layer provides scalar-templated residuals and explicit
linearizations for fixed-mesh Cartesian incompressible flow in two and three
dimensions. It covers constant density and viscosity, momentum mass,
advection, Newtonian stress, constant force per volume, and continuity.
Expressions receive their state, material data, and basis functions explicitly;
they do not access the legacy field or material globals.

The quadrature-point contributions follow the legacy sign conventions. With
velocity test function :math:`w`, pressure test function :math:`q`, velocity
:math:`u`, pressure :math:`p`, density :math:`\rho`, viscosity :math:`\mu`, and
momentum source :math:`f`, the unscaled terms are

.. math::

   R^{\mathrm{mass}}_a &= -w\rho\dot{u}_a, \\
   R^{\mathrm{advection}}_a &= -w\rho\sum_d u_d\partial_d u_a, \\
   R^{\mathrm{stress}}_a &= -\sum_d (\partial_d w)
       \left[\mu(\partial_d u_a + \partial_a u_d)-p\delta_{ad}\right], \\
   R^{\mathrm{source}}_a &= wf_a, \\
   R^{\mathrm{continuity}} &= q\sum_d \partial_d u_d.

Each equation's enabled terms and multipliers are applied separately, followed
by the quadrature weight and geometry determinant supplied by the caller.
Steady evaluation skips mass terms. The explicit velocity linearization uses
the time-derivative coefficient supplied by the existing solver, perturbing
both velocity and its time derivative consistently. The stress expression
uses the full symmetric gradient; dilational-viscosity corrections are outside
this initial subset.

These expressions are tested independently, including through Kokkos launches.

Q2/Q1 element evaluation
------------------------

The internal element evaluator supports 9-node quadrilaterals and 27-node
hexahedra with Q2 velocity, Q1 pressure on the corner nodes, and isoparametric
Q2 geometry. It preserves Goma's node ordering and uses the same 3-point
Gauss rule in each reference direction. Fixed geometry setup precomputes
physical basis gradients, quadrature positions, and weighted determinants.
Nonfinite geometry and nonpositive determinants at quadrature points invalidate
the element.

Each element is evaluated by one Kokkos team. Quadrature states are interpolated
into shared team storage before parallel evaluation of the local rows. Residual
requests allocate no local matrix; Jacobian requests omit residual storage;
combined requests reuse interpolation and quadrature data. Explicit
linearization uses the supplied time-derivative coefficient. The local ordering
is component-major velocity DOFs followed by the corner pressure DOFs; a future
host adapter must map this ordering to the existing solver DOFs.

Results contain one status per element. Invalid material data, nonfinite
solution fields, and nonfinite contributions are reported through that status.
An element with nonzero status must be discarded in its entirety, including
any partial values. Launches are asynchronous: callers must retain input and
output Views and keep the runtime alive until completion. Results can be
inspected after a fence or a host mirror copy.

Basis and quadrature values are compared with the production legacy routines.
Regular and distorted element Jacobians are checked against centered finite
differences, perturbing time derivatives consistently. These checks do not yet
establish parity with the full legacy assembly implementation.

Element constraints and global scatter
--------------------------------------

The internal Dirichlet kernel supports constant velocity and pressure data.
It replaces constrained element rows, preserves constrained columns, and adds
the full ``DIRICHLET_PENALTY`` for every incident element, as the legacy
implementation does. Hard-set conditions produce zero residual; residual-mode
conditions produce the penalty times the current value minus the prescribed
value. The host must impose hard-set values and zero their time derivatives
before gathering fields. The row-replacement kernel does not update the
solution or its time derivative.

A Tpetra scatter plan caches each element's owned row indices and local graph
entry offsets. Device kernels atomically add residual and Jacobian entries,
including columns for ghost DOFs. Only owned rows are assembled, so the caller
must provide every element incident on an owned row exactly once, including
ghost elements. The plan requires an owned-only active row map and a fixed,
complete graph containing all element couplings. Rebuild it after changing
the matrix, graph, or maps.

Plan setup and scatter are collective on the matrix communicator. Extent,
cache, phase, and element-status errors are rejected on all ranks before any
global writes. Scatter acquires writable Tpetra device values only for
Jacobian requests and fences before returning. It adds to existing values;
zeroing, scaling, and FE assembly phase transitions remain the caller's
responsibility. Jacobian requests require an open assembly or modification
phase. Residual-only requests leave matrix values untouched and may use a
closed matrix.

Tests cover constrained rows, shared nodes, ghost columns, repeated assembly,
empty ranks, and collective rejection of failed elements on one and two MPI
ranks with the Serial execution space. These checks do not yet qualify other
Kokkos execution spaces or establish end-to-end solver parity.

Host adaptation and preflight
-----------------------------

Preflight examines initialized problem, material, mesh-block, and boundary
metadata on every rank. It collects known incompatibilities and prints their
union once on the problem communicator's rank zero. Checks include coupled
matrix and material counts, active variables and equations, term flags and
multipliers, Q2/Q1 trial and test interpolation, the resolved geometry mapping,
constant Newtonian material models, stabilization, boundary types, and the
Tpetra/Stratimikos solver settings. An explicit constant nodal P condition is
required. Element-based pressure-datum handling remains unsupported.

The backend selection is broadcast with the other input settings. A rejected
setup cannot be solved, advanced, or retried; destroy its context before
creating the next one. The context retains its setup allocations until
``goma_destroy``, so cleanup follows the same path as an initialized problem.
Tests reject a Kokkos setup on two ranks and then solve a legacy problem in
the same library lifetime.

The host snapshot builder copies coordinates, constant material data, equation
term switches and multipliers, element DOF indices, global IDs, and nodal DBC
constraints. It accepts a host-side DOF resolver for the existing unknown map;
pressure DOFs are requested only at corner nodes. Snapshot data owns its
storage. The field-map adapter copies host vectors in bulk and gathers
component-major element fields on the device, without capturing legacy
pointers or invoking host callbacks in kernels. The caller must exchange
ghost values and impose hard-set solution values and time derivatives first.
Steady gathers do not need a time-derivative vector.

Solver routing remains unavailable even when the subset checks pass. The
snapshot and gather infrastructure is not yet connected to the nonlinear
assembly loop or the opaque context. Complete legacy assembly comparisons
and end-to-end steady/transient flow qualification are still required before
enabling Kokkos flow solves.
