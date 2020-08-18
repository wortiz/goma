#ifndef GOMA_MM_FILL_POISSON
#define GOMA_MM_FILL_POISSON

#include "goma.h"

extern int assemble_poisson(void);
extern void poisson_side_sin_bc(dbl func[DIM],
                         dbl d_func[DIM][MAX_VARIABLE_TYPES + MAX_CONC][MDE],
                         dbl alpha,
                         dbl beta,
                         dbl gamma,
                         dbl omega,
                         dbl zeta);

#endif // GOMA_MM_FILL_POISSON
