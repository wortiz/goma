*****************************
CFD-DEM Coupling (LAMMPS)
*****************************

Couples a transient volume-averaged Navier-Stokes (VANS) solve in *Goma* to a discrete
element (DEM) simulation in LAMMPS. Requires *Goma* built with ``-DENABLE_LAMMPS=ON``.
An installed shared LAMMPS library (``BUILD_SHARED_LIBS=on``, ``BUILD_MPI=on``,
``PKG_GRANULAR=on``, default ``LAMMPS_SIZES=smallbig``) is used if CMake finds it (set
``LAMMPS_DIR``). Otherwise, with ``ENABLE_LAMMPS_FETCHCONTENT=ON`` (default), LAMMPS is
downloaded from ``GOMA_LAMMPS_GIT_REPOSITORY`` (default ``https://github.com/lammps/lammps.git``)
at ``GOMA_LAMMPS_GIT_TAG`` (default ``stable_30Sep2026_update1``) and built as a static
library with the required settings. ``-DFETCHCONTENT_SOURCE_DIR_LAMMPS=<path>`` uses a local
LAMMPS checkout instead of downloading.

::

    CFD-DEM Coupling = {yes | no}
    LAMMPS Input File = <file_name>
    DEM Max Timestep = <float>
    DEM Drag Model = {DI_FELICE | STOKES}
    DEM Coupling Mode = {TWO_WAY | ONE_WAY | VOID_FRACTION}
    DEM Fix External ID = <id>
    DEM Minimum Void Fraction = <float>
    DEM 2D Depth = <float>
    DEM Drag Field Names = <name_x> <name_y> [name_z]

-----------------------
Description / Usage
-----------------------

The cards may appear anywhere in the input file; by convention they are placed in a
``CFD-DEM Coupling Specifications`` section after the Solver Specifications.

**CFD-DEM Coupling**
    ``yes`` replaces the transient solver with the coupled solver ``solve_problem_cfd_dem``.
    Requires ``Time integration = transient``, ``Use VANS Model = A | B`` (with
    ``Use AutoDiff Assembly = yes``), a single matrix, and the ``void_fraction`` equation.
    Default ``no``.

**LAMMPS Input File**
    LAMMPS input script read with ``lammps_file()`` on all ranks. Required. It must use
    ``atom_style sphere`` and must **not** define the fix external (see below). LAMMPS output
    goes to ``log.lammps``; screen output is suppressed.

**DEM Max Timestep**
    Largest DEM time step. Each accepted *Goma* step of size :math:`\Delta t` is followed by
    :math:`n_{sub} = \lceil \Delta t / \Delta t_{DEM,max} \rceil` LAMMPS steps of
    :math:`\Delta t / n_{sub}`, so LAMMPS follows *Goma*'s adaptive step. Without this card one
    DEM step is taken per *Goma* step (rarely what you want).

**DEM Drag Model**
    ``DI_FELICE`` (default), Di Felice (1994):

    .. math::

       \mathbf{F}_d = \tfrac{1}{2} C_d \rho_f \tfrac{\pi d^2}{4} \epsilon^2
       |\mathbf{u}-\mathbf{v}|(\mathbf{u}-\mathbf{v}) \epsilon^{-\chi},\quad
       C_d = \left(0.63 + \tfrac{4.8}{\sqrt{Re}}\right)^2,\quad
       \chi = 3.7 - 0.65 \exp\!\left(-\tfrac{(1.5-\log_{10} Re)^2}{2}\right)

    with :math:`Re = \rho_f \epsilon d |\mathbf{u}-\mathbf{v}| / \mu`. ``STOKES`` uses
    :math:`\mathbf{F}_d = 3\pi\mu d(\mathbf{u}-\mathbf{v})` without a void fraction correction
    and is intended for verification.

**DEM Coupling Mode**
    ``TWO_WAY`` (default): void fraction and drag on the particles, plus the reaction on the
    fluid. ``ONE_WAY``: void fraction and drag on the particles, no reaction on the fluid.
    ``VOID_FRACTION``: only the void fraction is passed; no fluid forces on the particles.

**DEM Fix External ID**
    ID of the ``fix external pf/callback 1 1`` that *Goma* creates in LAMMPS to apply fluid
    forces. Default ``cfd_drag``.

**DEM Minimum Void Fraction**
    Lower clip for the particle void fraction. Default ``0.35``. The number of clipped nodes
    is reported each step.

**DEM 2D Depth**
    Out-of-plane depth for a 2D *Goma* mesh coupled to a 3D LAMMPS simulation; nodal volumes
    are mesh areas times this depth. Required in that case. For a 2D LAMMPS simulation
    particles are disks and the depth is 1.

**DEM Drag Field Names**
    External fields that carry the fluid reaction force per unit volume. Defaults
    ``DEM_FX DEM_FY DEM_FZ``. For ``TWO_WAY`` each must be declared in the General
    Specifications with the ``CFD_DEM`` source keyword, e.g.::

        External Field = DEM_FX Q1 CFD_DEM
        External Field = DEM_FY Q1 CFD_DEM

    These fields start at zero, are filled by the coupler, and are written to the output
    EXODUS II file.

The void fraction unknown must be pinned with ``BC = FIX NS <all nodes> VOID_FRACTION 0``;
the coupler overwrites the ``VOID_FRACTION`` degrees of freedom (and their time derivative)
before every *Goma* step.

------------
Examples
------------

::

    External Field                   = DEM_FX Q1 CFD_DEM
    External Field                   = DEM_FY Q1 CFD_DEM
    ...
    Use AutoDiff Assembly            = yes
    Use VANS Model                   = A
    ...
    CFD-DEM Coupling                 = yes
    LAMMPS Input File                = in.dem
    DEM Max Timestep                 = 2.0e-5
    DEM Drag Model                   = DI_FELICE
    DEM Coupling Mode                = TWO_WAY
    DEM 2D Depth                     = 1.0
    ...
    BC = FIX NS 100 VOID_FRACTION 0

-------------------------
Technical Discussion
-------------------------

*Coupling scheme.* The coupling is staggered, fluid first. A *Goma* step from
:math:`t^n` to :math:`t^{n+1}` uses the void fraction and fluid reaction built from the
particles at :math:`t^n`. When the step is accepted, the converged fluid is sampled at each
particle, LAMMPS is advanced to :math:`t^{n+1}`, and the void fraction and reaction are
rebuilt from the new particle state. A rejected *Goma* step does not touch LAMMPS, so
nothing has to be rolled back. The void fraction therefore lags by one step.

*Drag sub-stepping.* At the start of each coupling interval the drag is linearized as
:math:`\mathbf{F}_d = \beta(\mathbf{u}_f - \mathbf{v})` with :math:`\beta` and the fluid
velocity :math:`\mathbf{u}_f` frozen at the particle. The LAMMPS callback evaluates the force
with the particle's current velocity on every DEM step, so particles relax correctly even when
the *Goma* step is much longer than the particle response time
:math:`\rho_p d^2 / 18\mu`. The fluid receives the drag averaged over the DEM sub-steps, which
makes the momentum exchange exact over each interval; the totals are printed every step.

*Force partition.* Following Zhou et al. (2010): for Model A the particle force is
:math:`\mathbf{F}_d - V_p \nabla p` and the fluid reaction is the drag only; for Model B the
particle force and fluid reaction are :math:`\mathbf{F}_d/\epsilon`. Viscous-stress, lift and
virtual mass forces are not included.

*Mapping.* Particles are located in the mesh using the element vertex map, and particle
volume and reaction force are deposited onto element vertices with the (non-negative) vertex
shape functions and lumped vertex volumes. Values at midside, face and center nodes of
quadratic elements are interpolated from the vertices. Particles should be smaller than the
mesh cells (a common rule of thumb is cells of at least three particle diameters).

*Parallel.* Both codes share ``MPI_COMM_WORLD``. In this version every rank gathers every
particle, which is simple and robust for moderate particle counts.

*Limitations.* Fixed (non-deforming) meshes of tri/quad or tet/hex elements; no level set,
augmenting conditions, XFEM, particle tracking, segregated solves or LIBRARY_MODE. LAMMPS
atom IDs must fit in a 32-bit ``int``.

--------------
References
--------------

R. Di Felice, The voidage function for fluid-particle interaction systems, Int. J. Multiphase
Flow 20 (1994) 153-159.

Z. Y. Zhou, S. B. Kuang, K. W. Chu, A. B. Yu, Discrete particle simulation of particle-fluid
flow: model formulations and their applicability, J. Fluid Mech. 661 (2010) 482-510.
