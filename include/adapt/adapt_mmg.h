#ifndef GOMA_ADAPT_MMG_H
#define GOMA_ADAPT_MMG_H

#ifdef __cplusplus
extern "C" {
#endif


#include "exo_struct.h"
#include "dpi.h"
#include "rf_io_structs.h"
#include "sl_util_structs.h"
void adapt_mesh_with_mmg(Exo_DB *exo,
                         Dpi *dpi,
                         struct Results_Description *rd,
                         int imtrx,
                         struct GomaLinearSolverData **ams,
                         double **x,
                         double **x_old,
                         double **x_older,
                         double **x_oldest,
                         double **x_update,
                         double **xdot,
                         double **xdot_old,
                         double **resid_vector,
                         double **scale,
                         double time1,
                         double theta,
                         double delta_t,
 double ***gvec_elem,
                         bool mapvar);

#ifdef __cplusplus
}   /* extern "C" */
#endif

#endif /* GOMA_ADAPT_MMG_H */