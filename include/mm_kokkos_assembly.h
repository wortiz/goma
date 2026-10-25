/* Internal C interface; available in ENABLE_KOKKOS_ASSEMBLY builds only. */
#ifndef GOMA_MM_KOKKOS_ASSEMBLY_H
#define GOMA_MM_KOKKOS_ASSEMBLY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct goma_kokkos_context goma_kokkos_context;

typedef enum {
  GOMA_KOKKOS_SUCCESS = 0,
  GOMA_KOKKOS_INVALID_ARGUMENT = -1,
  GOMA_KOKKOS_BAD_STATE = -2,
  GOMA_KOKKOS_RUNTIME_ERROR = -3,
  GOMA_KOKKOS_UNSUPPORTED = -4
} goma_kokkos_status;

typedef enum {
  GOMA_KOKKOS_RESIDUAL = 1,
  GOMA_KOKKOS_JACOBIAN = 2,
  GOMA_KOKKOS_RESIDUAL_AND_JACOBIAN = 3
} goma_kokkos_evaluation_request;

/* Host calls are serialized, as for the rest of the Goma library interface.
 * Initialize once per library lifetime, after MPI initialization. Borrow an
 * existing runtime; otherwise own it until runtime_finalize. Kokkos cannot be
 * initialized again after it has been finalized. */
goma_kokkos_status goma_kokkos_runtime_initialize(void);
/* Check before tearing down dependent libraries; does not change runtime state. */
goma_kokkos_status goma_kokkos_runtime_check_finalize(void);
/* Reject finalization while any assembly context still owns device state.
 * A borrowed runtime remains initialized and belongs to the host. */
goma_kokkos_status goma_kokkos_runtime_finalize(void);

/* Require an active runtime session and a NULL *context. Context creation
 * allocates device error storage; it does not enable flow assembly. */
goma_kokkos_status goma_kokkos_context_create(goma_kokkos_context **context);
/* Collective for a configured context. Add requested contributions to the
 * matrix and owned residual; zeroing, scaling, FE phase transitions, ghost
 * exchange, and hard-set field updates remain the caller's responsibility.
 * Residual ordering follows the Tpetra owned row map. Unrequested residual
 * pointers are ignored. Steady calls do not require velocity_dot. Calls finish
 * before returning; no host pointers are retained. Setup is in mm_kokkos_context.hpp. */
goma_kokkos_status goma_kokkos_context_assemble(goma_kokkos_context *context,
                                                const double *solution,
                                                size_t solution_size,
                                                const double *velocity_dot,
                                                size_t velocity_dot_size,
                                                double time_coefficient,
                                                goma_kokkos_evaluation_request request,
                                                double *owned_residual,
                                                size_t residual_size);
/* Error text is owned by the context and valid until its next operation or
 * destruction. Cleared after a successful operation; use the returned status
 * code to determine success. */
const char *goma_kokkos_context_last_error(const goma_kokkos_context *context);

/* Fence before releasing Views. Set *context to NULL; a NULL *context is OK.
 * The host must keep a borrowed runtime alive until after context destruction. */
goma_kokkos_status goma_kokkos_context_destroy(goma_kokkos_context **context);

#ifdef __cplusplus
}
#endif
#endif
