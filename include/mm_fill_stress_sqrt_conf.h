/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) 2022 Goma Developers, National Technology & Engineering   *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

#ifndef GOMA_MM_FILL_STRESS_SQRT_CONF_H
#define GOMA_MM_FILL_STRESS_SQRT_CONF_H
#include "mm_as_structs.h"
#include "std.h"

void compute_a_dot_b(dbl b[DIM][DIM],
                     dbl G[DIM][DIM],
                     dbl a_dot_b[DIM][DIM],
                     dbl d_a_dot_b_db[DIM][DIM][DIM][DIM],
                     dbl d_a_dot_b_dG[DIM][DIM][DIM][DIM]);

int sqrt_conf_source(int mode,
                     dbl b[DIM][DIM],
                     dbl source_term[DIM][DIM],
                     dbl d_source_term_db[DIM][DIM][DIM][DIM]);

int assemble_stress_sqrt_conf(dbl tt, /* parameter to vary time integration from
                                       * explicit (tt = 1) to implicit (tt = 0) */
                              dbl dt, /* current time step size */
                              PG_DATA *pg_data);

#endif /* GOMA_MM_FILL_STRESS_SQRT_CONF_H */
