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
/* File containing the many of Goma's global variables, moved for -fno-common */

#include "ac_particles.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_node_const.h"
#include "rf_solver.h"

/* Problem setup globals formerly defined here now live in goma_ctx->problem. */

int nnz_own; /* number of nonzeroes in the matrix that
              * belong to this processor
              */

int nnz_total; /* total number of nonzeroes in the matrix
                * that this processor sees, including the
                * entries for external unknowns that will
                * not be updated by this processor  */

int PRS_mat_ielem;

struct Boundary_Condition *BC_Types;
struct Rotation_Specs *ROT_Types;

NODE_INFO_STRUCT **Nodes = NULL;

struct elem_side_bc_struct ***First_Elem_Side_BC_Array;
struct elem_edge_bc_struct ***First_Elem_Edge_BC_Array;
