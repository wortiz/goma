#include "mm_assembly.h"

#include <stddef.h>
#include <string.h>

int goma_assembly_backend_parse(const char *name, goma_assembly_backend *backend) {
  if (name == NULL || backend == NULL) {
    return -1;
  }
  if (strcmp(name, "legacy") == 0) {
    *backend = GOMA_ASSEMBLY_LEGACY;
  } else if (strcmp(name, "kokkos") == 0) {
    *backend = GOMA_ASSEMBLY_KOKKOS;
  } else {
    return -1;
  }
  return 0;
}

const char *goma_assembly_backend_check(goma_assembly_backend backend, int autodiff) {
  if (backend == GOMA_ASSEMBLY_LEGACY) {
    return NULL;
  }
  if (backend != GOMA_ASSEMBLY_KOKKOS) {
    return "Invalid assembly backend";
  }
  (void)autodiff;
#ifndef GOMA_ENABLE_KOKKOS_ASSEMBLY
  return "Assembly Backend = kokkos requires a build with ENABLE_KOKKOS_ASSEMBLY=ON";
#else
  /* Complete-system capability checks run after problem setup on all ranks. */
  return NULL;
#endif
}
