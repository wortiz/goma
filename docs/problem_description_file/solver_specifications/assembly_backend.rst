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
Element geometry, Q2/Q1 interpolation, constraints, global matrix scatter,
complete-system capability checks, and end-to-end flow qualification remain
to be implemented. Expression availability does not enable the input option.
