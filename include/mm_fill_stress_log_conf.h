
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

#ifndef GOMA_MM_FILL_STRESS_LOG_CONF_H
#define GOMA_MM_FILL_STRESS_LOG_CONF_H
#include "mm_as_structs.h"
#include "std.h"

/*
 * This routine assembles the stress with a log-conformation tensor formulation.
 */
int assemble_stress_log_conf(dbl tt, dbl dt, PG_DATA *pg_data);

int assemble_stress_log_conf_transient(dbl tt, dbl dt, PG_DATA *pg_data);

void compute_exp_s(double s[DIM][DIM],
                   double exp_s[DIM][DIM],
                   double eig_values[DIM],
                   double R[DIM][DIM]);

void analytical_exp_s(double s[DIM][DIM],
                      double exp_s[DIM][DIM],
                      double eig_values[DIM],
                      double R[DIM][DIM],
                      double d_exp_s_ds[DIM][DIM][DIM][DIM]);

void compute_d_exp_s_ds(dbl s[DIM][DIM], // s - stress
                        dbl exp_s[DIM][DIM],
                        dbl d_exp_s_ds[DIM][DIM][DIM][DIM]);

#endif                                 /* GOMA_MM_FILL_STRESS_LOG_CONF_H */
