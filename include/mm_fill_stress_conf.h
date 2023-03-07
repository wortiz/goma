
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

#ifndef GOMA_MM_FILL_STRESS_CONF_H
#define GOMA_MM_FILL_STRESS_CONF_H
#include "mm_as_structs.h"
#include "std.h"

int assemble_stress_conf(dbl tt, /* parameter to vary time integration from
                                  * explicit (tt = 1) to implicit (tt = 0) */
                         dbl dt, /* current time step size */
                         PG_DATA *pg_data);

#endif /* GOMA_MM_FILL_STRESS_CONF_H */
