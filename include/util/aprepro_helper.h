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
#ifndef GOMA_APREPRO_HELPER_H
#define GOMA_APREPRO_HELPER_H

#ifdef GOMA_ENABLE_APREPRO_LIB

#ifdef __cplusplus
extern "C" {
#endif
#include "mm_as_structs.h"
#include "mm_eh.h"

goma_error aprepro_parse_file(char *infile, char *outfile);
goma_error aprepro_parse_goma_file(char *filename);
goma_error aprepro_parse_goma_single_var(char *filename, char *var, double val);
goma_error aprepro_parse_goma_augc(struct AC_Information *ac, double val);
#ifdef __cplusplus
}
#endif

#endif

#endif