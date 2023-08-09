/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) 2023 Goma Developers, National Technology & Engineering   *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/
#ifndef GOMA_MM_FILL_TURBULENT_H
#define GOMA_MM_FILL_TURBULENT_H

#ifdef EXTERN
#undef EXTERN
#endif

#ifdef GOMA_MM_FILL_TURBULENT_C
#define EXTERN
#else
#define EXTERN extern
#endif

#include "exo_struct.h"
#include "mm_as_structs.h"
#include "std.h"
#include "density.h"

int calc_vort_mag(dbl *vort_mag,
                         dbl omega[DIM][DIM],
                         dbl d_vort_mag_dv[DIM][MDE],
                         dbl d_vort_mag_dmesh[DIM][MDE]);
double turb_omega_wall_bc(void);

                       
typedef struct {
  dbl dk[MDE];
  dbl domega[MDE];
} blending_dependence_struct;

void calc_blending_functions(dbl rho,
                                    DENSITY_DEPENDENCE_STRUCT *d_rho,
                                    dbl *F1,
                                    dbl *F2,
                                    blending_dependence_struct *d_F1,
                                    blending_dependence_struct *d_F2);  
EXTERN int assemble_spalart_allmaras(dbl time_value, /* current time */
                                     dbl tt,         /* parameter to vary time integration from
                                                      explicit (tt = 1) to implicit (tt = 0)    */
                                     dbl dt,         /* current time step size                    */
                                     const PG_DATA *pg_data);

EXTERN int assemble_turb_k(dbl time_value, /* current time */
                                     dbl tt,         /* parameter to vary time integration from
                                                      explicit (tt = 1) to implicit (tt = 0)    */
                                     dbl dt,         /* current time step size                    */
                                     const PG_DATA *pg_data);

EXTERN int assemble_turb_omega(dbl time_value, /* current time */
                                     dbl tt,         /* parameter to vary time integration from
                                                      explicit (tt = 1) to implicit (tt = 0)    */
                                     dbl dt,         /* current time step size                    */
                                     const PG_DATA *pg_data);
#endif
