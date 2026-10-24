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

#ifndef GOMA_RESET_CHECK_H
#define GOMA_RESET_CHECK_H

#ifdef GOMA_CHECK_GLOBAL_RESET
#include <stdio.h>

/* Snapshot the writable data segments of the executable and libgoma */
void goma_reset_check_snapshot(void);

/* Print ranges that differ from the snapshot, returns the number of ranges */
int goma_reset_check_compare(FILE *out);

void goma_reset_check_free(void);
#endif

#endif /* GOMA_RESET_CHECK_H */
