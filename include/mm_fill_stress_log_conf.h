
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

#ifndef GOMA_MM_FILL_STRESS_LOG_CONF_H
#define GOMA_MM_FILL_STRESS_LOG_CONF_H
#include "el_elm.h"
#include "mm_as_structs.h"
#include "std.h"
void advective_decomposition(dbl grad_v[DIM][DIM],
                             dbl xi,
                             dbl s[DIM][DIM],
                             dbl R[DIM][DIM],
                             dbl R_T[DIM][DIM],
                             dbl eig_values[DIM],
                             dbl d_R[DIM][DIM][DIM][DIM],
                             dbl d_R_T[DIM][DIM][DIM][DIM],
                             dbl d_eig_values[DIM][DIM][DIM],
                             bool compute_jacobian_entries,
                             dbl advective_term[DIM][DIM],
                             dbl d_advective_term_ds[DIM][DIM][DIM][DIM]);

void source_term_logc(int mode,
                      dbl eig_values[DIM],
                      dbl R[DIM][DIM],
                      dbl R_T[DIM][DIM],
                      dbl d_eig_values[DIM][DIM][DIM],
                      dbl d_R[DIM][DIM][DIM][DIM],
                      dbl d_R_T[DIM][DIM][DIM][DIM],
                      dbl source_term[DIM][DIM],
                      dbl d_source_term[DIM][DIM][DIM][DIM]);

/*
 * This routine assembles the stress with a log-conformation tensor formulation.
 */
int assemble_stress_log_conf(dbl tt, dbl dt, PG_DATA *pg_data);

int assemble_stress_log_conf_transient(dbl tt, dbl dt, PG_DATA *pg_data);

void compute_exp_s(double s[DIM][DIM],
                   double exp_s[DIM][DIM],
                   double eig_values[DIM],
                   double R[DIM][DIM]);

void analytical_exp_s(dbl s[DIM][DIM],
                      dbl exp_s[DIM][DIM],
                      dbl eig_values[DIM],
                      dbl R[DIM][DIM],
                      dbl d_exp_s_ds[DIM][DIM][DIM][DIM],
                      dbl d_eig_values_ds[DIM][DIM][DIM],
                      dbl dR_ds[DIM][DIM][DIM][DIM]);

void compute_d_exp_s_ds(dbl s[DIM][DIM], // s - stress
                        dbl exp_s[DIM][DIM],
                        dbl d_exp_s_ds[DIM][DIM][DIM][DIM]);

#endif /* GOMA_MM_FILL_STRESS_LOG_CONF_H */
