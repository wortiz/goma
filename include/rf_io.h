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

/*
 * These are declarations for the global I/O varibles needed for
 * FEM problem specification
 *
 *
 * $Id: rf_io.h,v 5.3 2007-12-10 20:59:53 hkmoffa Exp $
 *
 *
 * Note: The associated constants are defined in "rf_io_const.h" and
 *       the definitions are defeind in rf_io_structs.h"
 */

#ifndef GOMA_RF_IO_H
#define GOMA_RF_IO_H

#include "rf_io_const.h" /* just in case it has not been done yet.
				  * -> We need this include dor this file */
#include "util/goma_context.h"

extern char *GomaPetscOptions;
extern int GomaPetscOptionsStrLen;

#ifdef MATRIX_DUMP
extern int Number_Jac_Dump; /* Number of jacobians to dump out
                             * If the value is negative, then the one
                             * jacobian, the -n'th jacobian, is dumped
                             * out */
#endif
extern int Num_Var_Init;         /* Number of variables to overwrite with
                                  * global initialization */
extern int Num_Var_Bound;        /* Number of variables to apply bounds */
extern int Num_Var_LS_Init;      /* number of variables to overwirte with
                                  * level set index initialization */
extern int Num_Var_External;     /* Number of external variables */
extern int Num_Var_External_pix; /* number of external variables (pixel only)*/
extern int Anneal_Mesh;          /* Flag specifying creation of a special exodus
                                  * file with coordinates adjusted to the
                                  * deformed coordinates (i.e. new displacements
                                  * are set to zero and mesh is deemed stress-free */

extern double Porous_liq_inventory; /*global variable for finite-insult boundary
                                     * condition
                                     */
extern double **Spec_source_inventory;
extern double *Spec_source_lumped_mass;
extern const char anneal_file[];

extern char front_scratch_directory[MAX_FNL]; /* def'd in main.c */

#endif
