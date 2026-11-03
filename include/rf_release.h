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
 * Teardown of the input deck, problem setup and mesh data owned by a
 * GomaContext (rf_release.c).
 */

#ifndef GOMA_RF_RELEASE_H
#define GOMA_RF_RELEASE_H

#include "util/goma_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Free everything a run allocated through ctx: post-processing requests,
 * boundary conditions and their element lists, data tables, unknown and
 * communication maps, nodes, element blocks, material file data, level set,
 * continuation and augmenting conditions, the Exodus and decomposition
 * databases, and the problem description structures (fem_structures_free()).
 * Installed as ctx->release_problem by pd_alloc(); ctx must be current.
 */
void goma_problem_release(GomaContext *ctx);

/* The pieces of goma_problem_release(), in the order it calls them. */
void post_processing_release(GomaPost *post);
void boundary_conditions_release(GomaContext *ctx);
void problem_setup_release(GomaContext *ctx);
void materials_release(GomaFem *fem);
void fem_input_release(GomaContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* GOMA_RF_RELEASE_H */
