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
//
/*
 * $Id: ac_particles.h,v 5.1 2007-09-18 18:53:40 prschun Exp $
 */

/*
 * Copyright (C) 2001 Sandia National Laboratories
 */

#ifndef GOMA_AC_PARTICLES_H
#define GOMA_AC_PARTICLES_H

#ifdef EXTERN
#undef EXTERN
#endif

#ifdef GOMA_AC_PARTICLES_C
#define EXTERN /* do nothing */
#endif

#include <stdio.h>

#include "ac_particles_types.h"
#include "ac_stability_util.h"
#include "el_elm.h"
#include "exo_struct.h"
#include "std.h"
#include "stdio.h"

#ifndef GOMA_AC_PARTICLES_C
#define EXTERN extern
#endif

#define MAX_DATA_REAL_VALUES 3
#define MAX_DATA_INT_VALUES  1

#define MAX_RANK 6 /* Used in the solve_NxN_system routine to avoid lots of malloc()'s. */

#define EXIT    1 /* For calling the dump() routine. */
#define NO_EXIT 0

/* This is the "right" way to do it, but the current Sun compilers
 * can't handle it, so I have all the more explicit call definisions
 * below. */
/*
#define dump(ec, p, ...) dump_fcn(__FILE__, __LINE__, ec, p, __VA_ARGS__)
*/
#define dump1(ec, p, x1)             dump_fcn(__FILE__, __LINE__, ec, p, x1)
#define dump2(ec, p, x1, x2)         dump_fcn(__FILE__, __LINE__, ec, p, x1, x2)
#define dump3(ec, p, x1, x2, x3)     dump_fcn(__FILE__, __LINE__, ec, p, x1, x2, x3)
#define dump4(ec, p, x1, x2, x3, x4) dump_fcn(__FILE__, __LINE__, ec, p, x1, x2, x3, x4)
#define dump10(ec, p, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) \
  dump_fcn(__FILE__, __LINE__, ec, p, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
#define dump20(ec, p, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17,  \
               x18, x19, x20)                                                                      \
  dump_fcn(__FILE__, __LINE__, ec, p, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, \
           x15, x16, x17, x18, x19, x20)

enum Particle_State_t { ACTIVE, DEAD, PROC_TRANSFER, TERMINATION_MARKER, GHOST };

typedef struct {
  dbl grad_V[DIM];
  dbl grad_Enorm[DIM];
} my_fv_old_t;

typedef struct p_t {
  dbl x[DIM];
  dbl x_old[DIM];
  dbl xi[DIM];
  dbl xi_old[DIM];
  dbl v[DIM];
  dbl v_old[DIM];
  dbl theta, theta_old;
  dbl phi, phi_old;
  dbl real_data[MAX_DATA_REAL_VALUES];
  dbl real_data_old[MAX_DATA_REAL_VALUES];
  dbl time, time_old;
  enum Particle_State_t state;
  int int_data[MAX_DATA_INT_VALUES];
  int int_data_old[MAX_DATA_INT_VALUES];
  int owning_elem_id;
  int output_sample_number;
  struct p_t *next, *last;
#ifdef PARALLEL
  dbl x_start[DIM];
  dbl x_end[DIM];
  int owning_proc_id;
#endif
} particle_t;

EXTERN void
initialize_particles(const Exo_DB *, dbl *const, dbl *const, dbl *const, dbl *const, dbl *const);

EXTERN int compute_particles(const Exo_DB *,
                             dbl *const,
                             dbl *const,
                             dbl *const,
                             dbl *const,
                             dbl *const,
                             const dbl,
                             const dbl,
                             const int);

EXTERN void rd_particle_specs /* mm_input_particles.c */
    (FILE *, char *);
#endif /* GOMA_AC_PARTICLES_H */
