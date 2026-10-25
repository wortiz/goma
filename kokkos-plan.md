# Incremental Kokkos assembly for Goma

## Summary

Introduce a separately selectable C++20 assembly implementation while retaining the legacy
implementation as the default and numerical reference. Migrate complete, supported configurations
incrementally; remove legacy capabilities only after demonstrated feature parity and regression
coverage.

The first milestone covers fixed-mesh, Cartesian, incompressible Newtonian Navier–Stokes:

- Steady and transient flow with constant density and viscosity.
- Q2 velocity/Q1 pressure on 2D quadrilaterals and 3D hexahedra.
- Standard Galerkin terms, constant body forces, velocity Dirichlet conditions, and a prescribed
  pressure datum.
- Explicit Jacobians, Tpetra matrices, and Stratimikos solves.
- Actual execution and correctness on Serial, Threads, OpenMP, and CUDA.

Sacado follows this milestone using the same residual expressions. Performance is measured initially;
a speedup is not an acceptance requirement.

## Architecture and interfaces

### Coexistence and configuration

- Add an optional `ENABLE_KOKKOS_ASSEMBLY` build feature, initially disabled. Enabled builds require
  C++20; legacy-only builds retain their existing toolchain support. Keep C11 unchanged.
- Add `Assembly Backend = legacy|kokkos`, defaulting to `legacy`. Retain the existing
  `Use AutoDiff Assembly` setting for Jacobian selection. Initially, Kokkos plus AutoDiff reports
  that support is unavailable; it becomes valid when the Sacado milestone lands.
- Preflight the complete coupled system before assembly. Report unsupported equations, material
  models, interpolation, stabilization, boundary conditions, and solver settings together. Never
  silently fall back or combine legacy and Kokkos contributions.
- Retain existing decks, nonlinear iteration, time integration, mesh input, and output. Select the
  new path above the serial element loop in [src/mm_fill.c](src/mm_fill.c).
- Expose an internal C-compatible interface with an opaque assembly context supporting
  setup/preflight, assembly, and destruction. Carry evaluation requests for residual-only,
  Jacobian-only, or both.

### Goma Expressions

Use small, statically dispatched, device-callable functors:

| Expression | Responsibility |
|---|---|
| Material models | Density, viscosity, and subsequently other constitutive properties |
| Momentum mass | Time derivative contribution |
| Momentum advection | Convective contribution |
| Momentum stress | Viscous stress and pressure contribution |
| Momentum source | Body-force contribution |
| Continuity | Incompressibility contribution |

Each expression provides scalar-templated residual evaluation. Explicit derivatives use a separate
linearization operation with the same inputs and term multipliers. Material models additionally
expose the tangents needed by explicit equation linearizations.

Compose these expressions inside one fused element kernel. Separate expressions support independent
testing and replacement; fusion shares interpolation, material evaluation, and quadrature data.
Separate term launches would add storage and launch overhead, while whole-equation functors would
make individual terms harder to migrate and test.

Residual-only evaluation uses `double` and allocates no derivative storage. Jacobian-only evaluation
avoids residual scatter. Combined evaluation shares intermediate quantities.

### Data and execution boundary

- Introduce owned C++ assembly state containing Kokkos Views for connectivity, geometry, basis data,
  degree-of-freedom maps, material parameters, constraints, and solution fields.
- Restrict legacy globals to the host adapter. Expressions receive explicit contexts and cannot
  access `fv`, `pd`, `mp`, `lec`, or `ad_fv`.
- Precompute fixed geometry, quadrature, and scatter mappings during setup. Refresh solution,
  time-derivative, and requested evaluation data on each assembly call.
- Group elements by topology and expression configuration. Start with one Kokkos team per element,
  parallelize local rows, and share quadrature data. Accumulate matrix and residual entries with
  double-precision atomics.
- Return device errors through status storage, then report them collectively on the host. No
  device-side allocation, exceptions, virtual dispatch, or legacy callbacks.
- Place new implementation under module-prefixed `mm_` components. Keep scheduling, expressions,
  and legacy adaptation separate.

## Numerical and solver integration

- Preserve legacy residual signs, stress conventions, quadrature, term multipliers,
  degree-of-freedom ordering, and transient linearization. In particular, pass the existing
  time-derivative coefficient into both explicit and eventual AD evaluation.
- Reproduce element-level Dirichlet row replacement, including `DIRICHLET_PENALTY`, hard-set versus
  residual behavior, and contributions from every incident element. Preserve constrained columns
  and existing solution-update semantics.
- Require an explicit pressure constraint in the initial supported configurations. Defer
  nullspace-based pressure handling.
- Preserve Goma’s current owned-row assembly convention and ghost-element coverage. Scatter only
  owned rows, include ghost columns, and verify partition-boundary contributions without duplicate
  accumulation.
- Extend the existing Tpetra adapter with device assembly access and cached local graph offsets.
  Respect Tpetra’s assembly state and host/device synchronization contracts; never call host
  row-insertion methods from kernels. The existing wrapper is in
  [src/linalg/sparse_matrix_tpetra.cpp](src/linalg/sparse_matrix_tpetra.cpp); Trilinos documents the
  relevant [FECrsMatrix interfaces](https://github.com/trilinos/Trilinos/blob/trilinos-release-16-2-0/packages/tpetra/core/src/Tpetra_FECrsMatrix_decl.hpp).
- Reuse Stratimikos solver configuration. Initially allow bulk transfers at the existing
  nonlinear-solver boundary and host-side scaling. Keep element evaluation and matrix scatter on
  the selected device.
- Add explicit Kokkos runtime ownership: borrow an initialized runtime, otherwise initialize it for
  Goma; destroy Views before finalization and never finalize a host-owned runtime. Support multiple
  Goma contexts sequentially within one runtime lifetime.

## Delivery sequence and migration policy

1. **Foundation and reference checks:** add backend selection, capability validation, C++20
   configuration, runtime ownership, assembly contexts, and test adapters that compare legacy and
   new element contributions.
2. **Serial flow implementation:** implement supported geometry, constant material expressions,
   equation terms, explicit Jacobians, constraints, and Tpetra integration. Establish steady and
   transient end-to-end agreement.
3. **Parallel backend qualification:** validate Threads, OpenMP, CUDA, and MPI; correct synchronization
   and ownership issues; record assembly, transfer, and solve timings separately.
4. **Sacado evaluation:** instantiate the existing residual expressions with bounded-storage Sacado
   forward AD. Seed element unknowns in derivative blocks, including their time-derivative
   dependence; extract ordinary double matrix entries. Start with blocks of 32 directions, avoiding
   per-point heap allocation. Retain explicit Jacobians as an independent selectable implementation.
5. **Feature expansion:** extend a documented capability matrix one feature at a time. Prioritize
   stabilization and variable material models, then additional element families, boundary
   conditions, moving meshes, and multiphysics couplings.

Use the Trilinos dependency’s Kokkos consistently. Qualify the repository’s current Trilinos 16.2
baseline with C++20 before considering a dependency upgrade. Maintain separate Serial, Threads,
OpenMP, and CUDA build configurations; Kokkos allows only one host parallel backend per build.
([Configuration guide](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html))

Keep C++23 as a later toolchain migration after the supported Kokkos, Trilinos, and CUDA combinations
are qualified.

Legacy retirement requires a completed capability matrix for the affected feature, passing numerical
and integration tests, documented limitations, and an explicit retirement change. Initial delivery
removes no legacy assembly.

## Current implementation and immediate next steps

The implementation through `33677081` includes backend selection and build checks,
Kokkos runtime ownership, device-callable flow expressions and explicit Jacobians,
Q2/Q1 geometry and element evaluation, Dirichlet row replacement, owned-row Tpetra
scatter, and host snapshot/gather helpers. Collective preflight reports known
incompatibilities after setup and rejects execution while solver routing and full
legacy parity remain unfinished. Legacy assembly remains the default.

Complete the remaining integration in this order:

1. **Opaque assembly context (implemented, 2026-10-25).** The context now owns
   copied host/device DOF maps, prepared geometry, constraints, material/expression
   data, and its Tpetra scatter plan. Typed snapshot setup is collective and
   validates metadata, geometry, owned-row coverage, and consistent setup options.
   A C-compatible evaluation entry point refreshes fields and handles residual,
   Jacobian, or combined requests without changing solver assembly phases. Failed
   internal setup is retryable; successful setup is immutable. Runtime finalization
   is blocked while contexts are alive. Serial and two-rank tests cover repeated
   evaluation, field/time-coefficient refresh, request modes, failed setup/input,
   teardown/recreation, and empty ranks. Actual legacy-data hookup and nonlinear
   solver routing remain the following steps.
2. **Wire actual Goma data (implemented, 2026-10-25).** One description builder
   (`describe_legacy_flow`) now serves both preflight and runtime setup, reading
   initialized globals without changing them. `setup_legacy_context` and the C
   entry point `goma_kokkos_context_setup_legacy` are collective on `goma_comm`:
   they snapshot through `Index_Solution`, the matrix's owned/ghost global-ID
   array, and `Nodes` DBC offsets, then require on every rank that the Tpetra
   owned row map equals the leading `n_rows` global IDs, that each local element
   (including ghost elements) appears exactly once, and that some rank owns a
   constrained pressure DOF. Local failures are joined before collective context
   setup, so a failure on one rank is reported on all and setup stays retryable.
   `goma_kokkos_context_assemble_legacy` accepts Goma's local `x`/`xdot` (owned
   plus external, already exchanged and hard-set) and adds requested contributions
   to the owned prefix of `resid_vector`, rejecting any matrix other than the one
   used at setup. Tests build Goma-layout node tables (owned nodes first,
   owner-contiguous global IDs, ghost elements on non-owners) and compare
   Index_Solution-backed assembly against an independent element reference on one,
   two, and three ranks, steady and transient, quads and hexes, including repeated
   updates, Jacobian-only and residual-only requests, residual and hard-set
   constraints, and one-rank failures. The adapter has not yet been driven by a
   real deck; that happens when step 3 routes `matrix_fill_full` through it.
3. **Route nonlinear assembly.** Select the prepared Kokkos context above the
   legacy element loop in `matrix_fill_full`, preserving residual-only and
   Jacobian-only requests, transient coefficients, matrix zeroing, owned residual
   ordering, scaling, and Stratimikos assembly phases. Attach context teardown to
   the Goma library lifecycle. Never silently fall back to legacy contributions.
4. **Establish legacy parity.** Compare constrained and unconstrained element and
   global contributions before scaling on regular/distorted quads and hexes, with
   shared nodes, ghost columns, and one/two MPI ranks. Run steady cavity,
   body-force, and transient-decay cases; compare norms, divergence, boundary
   values, and nonlinear convergence against legacy.
5. **Enable the supported subset.** Remove the execution gate only after the
   preceding integration and numerical checks pass; retain capability rejection
   for every unsupported configuration and keep the default backend unchanged.
6. **Qualify parallel execution and extend derivatives.** Run actual Threads,
   OpenMP, and CUDA configurations, measure assembly/transfers/solve separately,
   then add bounded-storage Sacado Jacobians using the existing residuals.

Existing checks cover Serial expression/element execution, one/two-rank scatter,
legacy-metadata context setup and evaluation, empty ranks, collective failure
before global writes, and rejected-setup recovery followed by a legacy solve. They do not yet qualify full Kokkos flow solves,
Threads/OpenMP/CUDA, or parity with the complete legacy assembly implementation.

## Validation and acceptance

- **Expression tests:** constant material values and tangents; individual equation terms; term
  switches and multipliers; pressure coupling; transient coefficients.
- **Derivative tests:** compare explicit Jacobians against centered finite differences on perturbed,
  nonzero element states. Perturb transient derivatives consistently. After Sacado lands, compare
  all three derivative implementations.
- **Element parity:** compare unconstrained and constrained residuals/Jacobians against legacy on
  regular and distorted quads and hexes. Use entrywise absolute-plus-relative tolerances, initially
  `1e-12 + 1e-10 × magnitude`; compare constraint rows separately.
- **Assembly parity:** compare global matrices and residuals before scaling, including shared nodes,
  ghost columns, constrained nodes on rank boundaries, and ranks with no local elements. Allow
  relative tolerance `1e-9` for changed summation order.
- **Flow cases:** steady cavity flow, body-force-driven flow, and transient viscous decay in 2D and
  3D. Check solution norms, divergence, boundary values, and nonlinear convergence against legacy;
  target normalized solution differences below `1e-8`.
- **Backend qualification:** run the supported cases on all four backends and with one and two MPI
  ranks. Record the actual execution space and verify CUDA kernels execute on a GPU; a skipped GPU
  job does not qualify CUDA.
- **Compatibility:** test unchanged legacy decks, builds without Sacado, unsupported-feature
  diagnostics, residual-only line searches, repeated contexts, and host-owned Kokkos runtime
  handling.
- Register tests with Catch2/CTest, add focused GUTS coverage, and start regression validation with
  `p_C_T_U`. Run the broad expensive suite before promotion to a default backend. Update Sphinx
  documentation and run formatting and strict documentation checks.

Initial defaults are one homogeneous fluid material, standard non-enriched Q2/Q1 elements, constant
prescribed boundary values, and existing time integration. Stabilization, ALE, axisymmetry,
additional physics, and surface boundary terms remain on the legacy path until explicitly added and
qualified.
