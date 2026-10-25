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

#ifdef __cplusplus
}
#endif
#endif
