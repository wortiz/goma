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
#ifndef GOMA_OMEGA_H_INTERFACE_H
#define GOMA_OMEGA_H_INTERFACE_H

#if defined(c_plusplus) || defined(__cplusplus)
extern "C" {
typedef struct Exodus_Database Exo_DB;
typedef struct Distributed_Processing_Information Dpi;
#else
#include "dpi.h"
#include "exo_struct.h"
#endif

void adapt_mesh_omega_h(struct GomaLinearSolverData **ams,
                        Exo_DB *exo,
                        Dpi *dpi,
                        double **x,
                        double **x_old,
                        double **x_older,
                        double **xdot,
                        double **xdot_old,
                        double **x_oldest,
                        double **resid_vector,
                        double **x_update,
                        double **scale,
                        int step);

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif
#endif

// vim: expandtab sw=2 ts=8
