#ifndef GOMA_MM_KOKKOS_LEGACY_H
#define GOMA_MM_KOKKOS_LEGACY_H

#include "exo_struct.h"
#include "mm_kokkos_assembly.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Collective on goma_comm; reports all known incompatibilities on rank zero.
 * Configurations outside the supported subset are rejected; accepted ones are
 * reported once on rank zero. */
goma_kokkos_status goma_kokkos_preflight_legacy(const Exo_DB *exo);

struct g_GomaSparseMatrix;
struct GomaLinearSolverData;

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

/* Collective on goma_comm. Replace the legacy element loop of matrix_fill_full for
 * the matrix in ams, creating and setting up its context on first use. Requests
 * follow af->Assemble_Residual and af->Assemble_Jacobian; the transient Jacobian
 * coefficient is (1 + 2 theta) / delta_t. The caller zeroes, exchanges, imposes,
 * and scales exactly as for legacy fills. Returns 0, or -1 on every rank after
 * reporting the failure; never falls back to legacy contributions. */
int goma_kokkos_matrix_fill(struct GomaLinearSolverData *ams,
                            const Exo_DB *exo,
                            const double *x,
                            const double *xdot,
                            double delta_t,
                            double theta,
                            double *resid_vector);

/* Destroy the fill context bound to matrix, or any fill context when matrix is
 * NULL. Call before the matrix and its graph are destroyed. Local; requires the
 * Kokkos runtime still to be active. */
void goma_kokkos_release_matrix(const struct g_GomaSparseMatrix *matrix);

#ifdef __cplusplus
}
#endif
#endif
