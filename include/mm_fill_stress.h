
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

#ifndef GOMA_MM_FILL_STRESS_H
#define GOMA_MM_FILL_STRESS_H
#include "mm_as_structs.h"
#include "std.h"
bool is_evss_f_model(int model);
bool is_log_c_model(int model);

void alloc_fv_log_c(struct Field_Variables *fv);

void free_fv_log_c(struct Field_Variables *fv);

void load_fv_log_c(struct Field_Variables *fv,
                   struct Element_Stiffness_Pointers *esp,
                   bool compute_jacobian);

void ve_stress_term(dbl mu,
                    dbl mus,
                    VISCOSITY_DEPENDENCE_STRUCT *d_mu,
                    VISCOSITY_DEPENDENCE_STRUCT *d_mus,
                    dbl stress[DIM][DIM],
                    STRESS_DEPENDENCE_STRUCT *d_stress);

void momentum_ve_stress_term(dbl mu,
                             dbl mus,
                             dbl mu_over_mu_num,
                             VISCOSITY_DEPENDENCE_STRUCT *d_mu,
                             VISCOSITY_DEPENDENCE_STRUCT *d_mus,
                             dbl d_mun_dS[MAX_MODES][DIM][DIM][MDE],
                             dbl d_mun_dG[DIM][DIM][MDE],
                             dbl stress[DIM][DIM],
                             STRESS_DEPENDENCE_STRUCT *d_stress);

int assemble_stress_fortin(dbl,        /* tt - parm to vary time integration from
                                        * explicit (tt = 1) to implicit (tt = 0)    */
                           dbl,        /* dt - current time step size               */
                           PG_DATA *); /* Petrov-Galerkin Data (SUPG) */

#endif /* GOMA_MM_FILL_STRESS_H */
