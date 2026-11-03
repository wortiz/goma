/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) Goma Developers, National Technology & Engineering        *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

/* mm_mp.h -- include file for structure pointing to material properties
 *
 * Created:	Sat Mar 19 15:51:22 MST 1994 pasacki@sandia.gov
 *
 * Revised:
 *
 * $Id: mm_mp.h,v 5.3 2008-01-25 14:58:26 hkmoffa Exp $
 */
#ifndef GOMA_MM_MP_H
#define GOMA_MM_MP_H

#include "mm_mp_structs.h"
#include "util/goma_context.h" /* material pointers live in goma_ctx->fem */

/*
 *  Some Prototypes
 *
 */
/*      mm_matrl.c */

GOMA_EXTERN void matrl_prop_print(MATRL_PROP_STRUCT *, int);
GOMA_EXTERN int goma_mat_prop_init(MATRL_PROP_STRUCT *, int, PROBLEM_DESCRIPTION_STRUCT *);
GOMA_EXTERN double calc_density(MATRL_PROP_STRUCT *, int, PROPERTYJAC_STRUCT *, double time);
GOMA_EXTERN double calc_concentration(MATRL_PROP_STRUCT *, int, PROPERTYJAC_STRUCT *);
GOMA_EXTERN void load_properties(MATRL_PROP_STRUCT *, double);

#endif
