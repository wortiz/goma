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
#ifndef GOMA_ENABLE_KOKKOS_ASSEMBLY
  (void)autodiff;
  return "Assembly Backend = kokkos requires a build with ENABLE_KOKKOS_ASSEMBLY=ON";
#else
  if (autodiff) {
    return "Assembly Backend = kokkos does not yet support Use AutoDiff Assembly = yes";
  }
  return "Assembly Backend = kokkos is not yet available: flow assembly kernels are not "
         "implemented";
#endif
}
