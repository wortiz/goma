#ifndef GOMA_COMPUTE_LAGGED_VARIABLES_H
#define GOMA_COMPUTE_LAGGED_VARIABLES_H

#ifdef __cplusplus
extern "C" {
#endif

#include "exo_struct.h"
#include "dpi.h"

struct Lagged_Variables {
  double *lagged_variables[MAX_PROB_VAR];
  int local_count;
  int global_count;
  int local_rows;
  int *local_node_to_lagged;
  double *exchange_lagged[MAX_PROB_VAR];
  int *mapping_index;
  int *index;
};

void setup_lagged_variables(Exo_DB *exo, Dpi *dpi, struct Lagged_Variables *lv);
int compute_lagged_variables(Exo_DB * exo, Dpi *dpi, dbl * x, dbl *x_old, dbl *x_older, dbl *xdot, dbl *xdot_old, struct Lagged_Variables *lv);

#ifdef __cplusplus
}
#endif

#endif // GOMA_COMPUTE_LAGGED_VARIABLES_H