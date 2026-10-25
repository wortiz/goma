/* Internal C interface; available in ENABLE_KOKKOS_ASSEMBLY builds only. */
#ifndef GOMA_MM_KOKKOS_ASSEMBLY_H
#define GOMA_MM_KOKKOS_ASSEMBLY_H

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
/* Fence before releasing Views. Set *context to NULL; a NULL *context is OK.
 * The host must keep a borrowed runtime alive until after context destruction. */
goma_kokkos_status goma_kokkos_context_destroy(goma_kokkos_context **context);

#ifdef __cplusplus
}
#endif
#endif
