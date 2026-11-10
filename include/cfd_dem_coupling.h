/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) 2026 Goma Developers, National Technology & Engineering   *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

/*
 * CFD-DEM coupling between Goma (volume averaged Navier-Stokes) and LAMMPS (DEM).
 *
 * Staggered, fluid-first scheme (see solve_problem_cfd_dem in rf_solve_cfd_dem.c):
 *
 *   goma  : t^n -> t^{n+1} with void fraction eps^n and particle reaction force F^n
 *   accept: drag on particles from u^{n+1}, LAMMPS runs n_sub steps to t^{n+1}
 *   then  : eps^{n+1}, F^{n+1} rebuilt from the new particle state
 *
 * A rejected goma step never touches LAMMPS, so nothing needs to be rolled back.
 */

#ifndef GOMA_CFD_DEM_COUPLING_H
#define GOMA_CFD_DEM_COUPLING_H

#include <stdio.h>
#include <string.h>

#include "dpi.h"
#include "el_elm.h"
#include "exo_struct.h"
#include "rf_io_const.h"
#include "std.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Drag correlations */
#define CFD_DEM_DRAG_DI_FELICE 1
#define CFD_DEM_DRAG_STOKES    2

/* Coupling modes */
#define CFD_DEM_COUPLE_VOID_FRACTION 1 /* particles -> eps only, no drag anywhere   */
#define CFD_DEM_COUPLE_ONE_WAY       2 /* eps + drag on particles, no fluid reaction */
#define CFD_DEM_COUPLE_TWO_WAY       3 /* eps + drag on particles + fluid reaction   */

#define CFD_DEM_NAME_LEN 64

/* External fields filled by the coupler, e.g. "External Field = DEM_FX Q1 CFD_DEM".
 * They are not read from, decomposed, or renamed as files. */
#define CFD_DEM_FIELD_SOURCE "CFD_DEM"
static inline int cfd_dem_is_coupler_field(const char *file_nm) {
  return strcmp(file_nm, CFD_DEM_FIELD_SOURCE) == 0;
}

struct Cfd_Dem_Specs {
  int enabled;                          /* "CFD-DEM Coupling = yes"                     */
  char lammps_input[MAX_FNL];           /* "LAMMPS Input File"                          */
  double dem_max_dt;                    /* "DEM Max Timestep", n_sub = ceil(dt/dem_max) */
  int drag_model;                       /* "DEM Drag Model"                             */
  int coupling_mode;                    /* "DEM Coupling Mode"                          */
  char fix_id[CFD_DEM_NAME_LEN];        /* "DEM Fix External ID"                        */
  double eps_min;                       /* "DEM Minimum Void Fraction"                  */
  double depth_2d;                      /* "DEM 2D Depth" for 2D goma + 3D particles    */
  char drag_field_names[DIM][CFD_DEM_NAME_LEN]; /* "DEM Drag Field Names"               */
  int drag_field_index[DIM]; /* resolved efv index of each drag field, -1 if unused     */
};

/* Statically initialized with defaults in cfd_dem_coupling.c; broadcast in noahs_ark. */
extern struct Cfd_Dem_Specs *cfd_dem;

/* Read the optional "CFD-DEM Coupling Specifications" section (rank 0, mm_input.c). */
void rd_cfd_dem_specs(FILE *ifp, char *input);

/* Coupled transient solver, used instead of solve_problem when cfd_dem->enabled. */
void solve_problem_cfd_dem(Exo_DB *exo, Dpi *dpi);

/* Coupler lifecycle, implemented in cfd_dem_coupling.c (require GOMA_ENABLE_LAMMPS). */
int cfd_dem_init(Exo_DB *exo, Dpi *dpi, double *x, double time);
/* Write void fraction into x (and a consistent xdot) at the VOID_FRACTION dofs. */
int cfd_dem_apply_void_fraction(
    double *x, double *x_old, double *xdot, double *xdot_old, double delta_t, double theta);
/* After an accepted goma step: drag -> LAMMPS run -> regather -> rebuild eps and source. */
int cfd_dem_advance(double *x,
                    double *x_old,
                    double *xdot,
                    double *xdot_old,
                    double time_old,
                    double time_new,
                    Exo_DB *exo,
                    Dpi *dpi);
void cfd_dem_finalize(void);

#ifdef __cplusplus
}
#endif

#endif /* GOMA_CFD_DEM_COUPLING_H */
