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

The Kokkos assembly path is still under development. Selecting ``kokkos``
reports an error before assembly, including in an enabled build. It never
falls back to legacy assembly. Kokkos with ``Use AutoDiff Assembly = yes`` is
also unavailable.

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

The evaluator currently produces unconstrained local element contributions.
Constraints, global matrix scatter, complete-system capability checks, and
end-to-end flow qualification remain to be implemented. The input option stays
unavailable until those integration checks pass.
