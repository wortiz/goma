#ifndef GOMA_AD_LEVEL_SET_H
#define GOMA_AD_LEVEL_SET_H

#ifdef __cplusplus
#include "ad/structs.h"
struct AD_Level_Set_Interface {
  /* Flag indicating if we're in the interfacial region. */
  int near;

  /* Half the interfacial thickness (alpha = 0.5 * width) */
  ADType alpha;

  /* Heaviside function; smooth form of: H=0 for F<0, H=1 for F>0 */
  ADType H;
  ADType dH;

  /* Heaviside function as above, but evaluated using FEM basis functions */
  ADType Hn;
  ADType Hn_old;
  ADType gradHn[DIM];
  ADType gradHn_old[DIM];

  /*
   * Delta function: smooth form of: delta(F) = 1 for F=0, =0 for F != 0
   * N.B. This delta has a correction for cases where F is not a pure
   * distance function.
   */
  ADType delta;
  ADType delta_max;

  /*
   * Normal vector: typically normal = grad(F); here we use normal =
   * grad(F) / |grad(F)| to be safe.
   */
  ADType normal[DIM];

  /* Magnitude of grad_F. */
  ADType gfmag;

  /* Magnitude of grad_F inverse. */
  ADType gfmaginv;
};

extern std::unique_ptr<AD_Level_Set_Interface> ad_lsi;

int ad_load_lsi(const double width) ;
extern "C" {
#endif

#include "exo_struct.h"
#include "mm_as_structs.h"

int ad_assemble_fill(double tt,
                     double dt,
                     const PG_DATA *pg_data,
                     const int applied_eqn,
                     double xi[3],
                     Exo_DB *exo,
                     double time,
                     struct LS_Mass_Lumped_Penalty *mass_lumped_penalty);

#ifdef __cplusplus
}
#endif

#endif // GOMA_AD_LEVEL_SET_H