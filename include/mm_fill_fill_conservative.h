#ifndef MM_FILL_FILL_CONSERVATIVE_H
#define MM_FILL_FILL_CONSERVATIVE_H

#include "mm_as_structs.h"

double heaviside_smooth(double field, double *d_heaviside_dfield, double alpha);

int
assemble_eikonal(int dim,
                 double tt,
                 double dt,
                 PG_DATA *pg_data);

int
assemble_heaviside_smooth(int dim,
                 double tt,
                 double dt,
                 PG_DATA *pg_data);

int
assemble_fill_prime(int dim,
                 double tt,
                 double dt,
                 PG_DATA *pg_data);

int
assemble_heaviside_projection(int dim,
                 double tt,
                 double dt,
                 PG_DATA *pg_data);

#endif
