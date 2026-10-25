/* Opt-in comparison of legacy and Kokkos assembly for qualification runs. */
#ifndef GOMA_MM_KOKKOS_PARITY_H
#define GOMA_MM_KOKKOS_PARITY_H

#include "exo_struct.h"
#ifdef __cplusplus
extern "C" {
#endif

struct GomaLinearSolverData;

/* Collective on goma_comm; nonzero when rank zero sets GOMA_KOKKOS_PARITY=1. */
int goma_kokkos_parity_enabled(void);

/* Run the legacy element loop into the caller's matrix and residual. */
typedef int (*goma_kokkos_reference_fill)(void *data);

/* Collective on goma_comm. Run the legacy reference fill and the Kokkos fill
 * from the same matrix and residual state, compare their owned residual and
 * matrix contributions before scaling, and leave the Kokkos values for the
 * solver. Entries agree when |kokkos - legacy| <= 1e-9 max(|kokkos|, |legacy|)
 * + 1e-12 (s + 1e-3 S). For matrix entries s is the row's largest legacy entry;
 * for residuals it is the row's sum of |J x| terms plus its contribution (cached
 * from the last Jacobian fill for residual-only requests). S is the largest s on
 * any rank, a floor for rows whose own terms vanish. Constrained rows are reported
 * separately. Rank zero prints one summary per fill; ranks print their worst
 * failing entries. Returns -1 on every rank if either fill fails or any entry
 * is outside tolerance. */
int goma_kokkos_parity_fill(struct GomaLinearSolverData *ams,
                            const Exo_DB *exo,
                            const double *x,
                            const double *xdot,
                            double delta_t,
                            double theta,
                            double *resid_vector,
                            goma_kokkos_reference_fill legacy,
                            void *data);

#ifdef __cplusplus
}
#endif
#endif
