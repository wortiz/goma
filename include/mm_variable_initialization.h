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

/* Variable initialization records, kept free of other Goma headers. */

#ifndef GOMA_MM_VARIABLE_INITIALIZATION_H
#define GOMA_MM_VARIABLE_INITIALIZATION_H

struct Variable_Initialization {
  int var;
  int ktype;
  double init_val;
  double init_val_min;
  double init_val_max;
  double init_val_minus; /* Value in negative LS phase */
  double init_val_plus;  /* Value in positive LS phase */
  int slave_block;       /* this is set TRUE in order that this initialization value
                           is not applied to nodes that are shared with adjacent blocks */
  int len_u_pars;
  double *u_pars;
};

#endif /* GOMA_MM_VARIABLE_INITIALIZATION_H */
