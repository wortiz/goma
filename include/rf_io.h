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

#ifdef MATRIX_DUMP
extern int Number_Jac_Dump; /* Number of jacobians to dump out
                             * If the value is negative, then the one
                             * jacobian, the -n'th jacobian, is dumped
                             * out */
#endif

extern const char anneal_file[];

#endif
