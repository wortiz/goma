/* Experimental assembly backend selection. */
#ifndef GOMA_MM_ASSEMBLY_H
#define GOMA_MM_ASSEMBLY_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { GOMA_ASSEMBLY_LEGACY = 0, GOMA_ASSEMBLY_KOKKOS = 1 } goma_assembly_backend;

/* Return zero on success; leave backend unchanged for an invalid name. */
int goma_assembly_backend_parse(const char *name, goma_assembly_backend *backend);
/* Check build availability only. Physics and execution checks run after setup. */
const char *goma_assembly_backend_check(goma_assembly_backend backend, int autodiff);

#ifdef __cplusplus
}
#endif
#endif
