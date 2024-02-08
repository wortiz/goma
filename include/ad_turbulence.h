#ifndef GOMA_AD_TURBULENCE_H
#define GOMA_AD_TURBULENCE_H

#ifdef GOMA_ENABLE_SACADO

#ifdef __cplusplus
extern "C" {
#endif


#include "std.h"
#include "mm_as_structs.h"
#include "mm_mp_structs.h"
#include "mm_fill_stabilization.h"


void ad_tau_momentum_shakib(momentum_tau_terms *tau_terms, int dim, dbl dt, int pspg_scale);

int ad_assemble_turb_k(dbl time_value, /* current time */
                               dbl tt,         /* parameter to vary time integration from
                                                  explicit (tt = 1) to implicit (tt = 0)    */
                               dbl dt,         /* current time step size                    */
                               const PG_DATA *pg_data);

void ad_sa_wall_func(double func[DIM],
                                 double d_func[DIM][MAX_VARIABLE_TYPES + MAX_CONC][MDE]);
dbl ad_turb_k_omega_sst_viscosity(VISCOSITY_DEPENDENCE_STRUCT *d_mu);

int ad_assemble_turb_omega(dbl time_value, /* current time */
                                   dbl tt,         /* parameter to vary time integration from
                                                      explicit (tt = 1) to implicit (tt = 0)    */
                                   dbl dt,         /* current time step size                    */
                                   const PG_DATA *pg_data);
dbl ad_sa_viscosity(struct Generalized_Newtonian *gn_local,
                               VISCOSITY_DEPENDENCE_STRUCT *d_mu);
void fill_ad_field_variables();
int ad_assemble_spalart_allmaras(dbl time_value, /* current time */
                              dbl tt,         /* parameter to vary time integration from
                                                 explicit (tt = 1) to implicit (tt = 0)    */
                              dbl dt,         /* current time step size                    */
                              const PG_DATA *pg_data);
#ifdef __cplusplus
}
#endif

#endif

#endif // GOMA_AD_TURBULENCE_H