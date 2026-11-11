# CFD-DEM lid driven cavity with a particle lattice (2D, one-way)

Goma solves the volume averaged Navier-Stokes equations (VANS model A) in a unit square
cavity with the lid moving at U = 1 (rho = 1, mu = 0.01, Re = 100). LAMMPS carries a
15 x 15 square lattice of disks (d = 0.015, spacing 0.0625, areal density 2.5) that start at
rest and are dragged around by the flow.

A finer parallel version (100 x 100 mesh, 5329 smaller disks, 8 MPI ranks) is in
`../ldc_particles_fine`.

`DEM Coupling Mode = ONE_WAY`: the fluid sees the particle void fraction (through the VANS
terms), the particles feel Di Felice drag plus `-V_p grad p`, and the fluid feels no reaction
force. Switching to `TWO_WAY` also needs the `External Field = DEM_FX/DEM_FY Q1 CFD_DEM`
cards (see `docs/problem_description_file/cfd_dem_coupling/cfd_dem_coupling_cards.rst`).

| File | Purpose |
|---|---|
| `input` | Goma problem: transient, adaptive dt up to 0.02, trapezoid rule, t = 0 to 10 |
| `fluid.mat` | Newtonian fluid properties |
| `in.dem` | LAMMPS: 2D granular disks, Hookean contacts, granular walls on the cavity edges |
| `ldc_dem.exoII` | 20 x 20 QUAD9 mesh (h = 0.05, about 3.3 d); node sets 1 bottom, 2 right, 3 lid, 4 left, 100 all nodes; side sets 1-4 on the same walls |
| `make_mesh.py` | Regenerates the mesh: `PYTHONPATH=$SEACAS/lib python3 make_mesh.py --n 20` |
| `plot_results.py` | Plots `out.exoII` + `dump.particles` (needs numpy, matplotlib, netCDF4) |
| `figures/` | Example results of this case made with `plot_results.py` (see below) |

Run (Goma built with `-DENABLE_LAMMPS=ON`):

    goma -i input

Outputs `out.exoII` (velocity, pressure, `VOID_FRACTION`, stream function), LAMMPS
`log.lammps`, and `dump.particles` (positions and velocities every 200 DEM steps, about 0.02
time units). `out.exoII` is about 20 MB and `dump.particles` about 6 MB.
The full run takes about 3 minutes on one core.

Visualize (run in this directory after the simulation):

    pip install numpy matplotlib netCDF4 imageio-ffmpeg   # or prefix with `uv run`
    python3 plot_results.py                       # |u|, streamlines, particles at the last time
    python3 plot_results.py --time 1 5 10 --trails 10
    python3 plot_results.py --field VOID_FRACTION # any nodal variable (--list shows them)
    python3 plot_results.py --animate cavity.mp4 --trails 10 --fps 20   # or .gif
    python3 plot_results.py --history history.png # particle speed, min/mean void fraction

Goma writes a frame every `Printing Frequency` steps (2, about every 0.04 time units) and
LAMMPS every 200 DEM steps (about 0.02). Each fluid frame is drawn with the nearest particle
frame; the title shows both times when they differ. Raise both intervals for smaller output.

Example results (`figures/`, trapezoid rule, made with the commands below):

| File | Shows |
|---|---|
| `velocity_t001.000.png`, `velocity_t005.000.png`, `velocity_t010.000.png` | Velocity magnitude, streamlines, particles with their last 0.3 time units of path |
| `void_fraction_t5.png` | Void fraction at t = 5 |
| `history.png` | Mean and max particle speed; min and mean nodal void fraction |
| `cavity.mp4` | 264 frames from t = 0 to 10 (13 s at 20 fps, 720 x 640, H.264) |

    python3 plot_results.py --time 1 5 10 --trails 15 --out figures/velocity.png
    python3 plot_results.py --field VOID_FRACTION --time 5 --out figures/void_fraction_t5.png
    python3 plot_results.py --history figures/history.png
    python3 plot_results.py --animate figures/cavity.mp4 --trails 10 --fps 20 --dpi 100

Notes

- The trapezoid rule (`Time step parameter = 0.5`) and backward Euler (`0.0`) both run to
  t = 10 with no rejected steps, in about 3.3 and 2.7 minutes. Their fluid velocities agree
  to within 1-3 % RMS.
- The smallest nodal void fraction dips to about 0.42 near t = 1, while particles crowd the
  top-left corner. After that it stays between 0.6 and 0.85, above the 0.35 clip. Wall and
  corner nodes have small lumped volumes, so a disk pressed into a corner lowers ε there
  the most.
- The mean particle speed levels off near 0.2, but the maximum spikes up to about 5, five
  times the lid speed. Every sample above 1.5 is a disk leaving the top-left corner, where
  the lid meets the wall. The likely causes are the singular pressure gradient at that corner
  (model A applies -V_p grad p to the particles) or rebound from the soft contacts of disks
  packed into the corner. Neither has been confirmed.
