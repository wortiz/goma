#ifndef GOMA_MM_KOKKOS_LEGACY_H
#define GOMA_MM_KOKKOS_LEGACY_H

#include "exo_struct.h"
#include "mm_kokkos_assembly.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Collective on goma_comm; reports all known incompatibilities on rank zero.
 * Solver routing remains gated even if the configuration passes the subset checks. */
goma_kokkos_status goma_kokkos_preflight_legacy(const Exo_DB *exo);

struct g_GomaSparseMatrix;

/* Collective on goma_comm. Configure an unconfigured context from the initialized
 * mesh, Index_Solution, Nodes DBC metadata, and the matrix's owned/ghost global IDs.
 * The matrix graph, global IDs, and node metadata must outlive the context. */
goma_kokkos_status goma_kokkos_context_setup_legacy(goma_kokkos_context *context,
                                                    const Exo_DB *exo,
                                                    struct g_GomaSparseMatrix *matrix);

/* Collective on goma_comm, with the matrix used for setup. x and xdot are local
 * owned-plus-external vectors (n_cols entries) after ghost exchange and hard-set
 * imposition; xdot may be NULL for steady problems. Requested residual contributions
 * are added to the owned prefix of resid_vector (n_rows entries); external entries
 * are unchanged. See goma_kokkos_context_assemble for matrix and phase contracts. */
goma_kokkos_status goma_kokkos_context_assemble_legacy(goma_kokkos_context *context,
                                                       struct g_GomaSparseMatrix *matrix,
                                                       const double *x,
                                                       const double *xdot,
                                                       double time_coefficient,
                                                       goma_kokkos_evaluation_request request,
                                                       double *resid_vector);

#ifdef __cplusplus
}
#endif
#endif
