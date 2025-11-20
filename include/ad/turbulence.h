#ifndef GOMA_AD_TURBULENCE_H
#define GOMA_AD_TURBULENCE_H

#include "ad/structs.h"

#ifdef GOMA_ENABLE_SACADO

#ifdef __cplusplus
#include <Sacado.hpp>
extern "C" {
#include "el_elm.h"
#include "mm_mp_const.h"
#include "std.h"
}
void ad_supg_tau_shakib(ADType &supg_tau, int dim, dbl dt, ADType diffusivity, int interp_eqn);
int ad_calc_shearrate(ADType &gammadot,            /* strain rate invariant */
                      ADType gamma_dot[DIM][DIM]); /* strain rate tensor */

void ad_only_tau_momentum_shakib(ADType &tau, int dim, dbl dt, int pspg_scale);
ADType ad_sa_viscosity(struct Generalized_Newtonian *gn_local);
ADType ad_only_turb_k_omega_viscosity(void);
void compute_sst_blending(ADType &F1, ADType &F2);
ADType sst_viscosity(const ADType &Omega, const ADType &F2);
extern "C" {
#endif

#include "mm_as_structs.h"
#include "mm_fill_stabilization.h"
#include "mm_mp_structs.h"
#include "std.h"

void ad_tau_momentum_shakib(momentum_tau_terms *tau_terms, int dim, dbl dt, int pspg_scale);

int ad_assemble_turb_k(dbl time_value, /* current time */
                       dbl tt,         /* parameter to vary time integration from
                                          explicit (tt = 1) to implicit (tt = 0)    */
                       dbl dt,         /* current time step size                    */
                       const PG_DATA *pg_data);

void ad_sa_wall_func(double func[DIM], double d_func[DIM][MAX_VARIABLE_TYPES + MAX_CONC][MDE]);
dbl ad_turb_k_omega_sst_viscosity(VISCOSITY_DEPENDENCE_STRUCT *d_mu);

int ad_assemble_turb_omega(dbl time_value, /* current time */
                           dbl tt,         /* parameter to vary time integration from
                                              explicit (tt = 1) to implicit (tt = 0)    */
                           dbl dt,         /* current time step size                    */
                           const PG_DATA *pg_data);
dbl ad_sa_viscosity(struct Generalized_Newtonian *gn_local, VISCOSITY_DEPENDENCE_STRUCT *d_mu);
void fill_ad_field_variables();
int ad_assemble_spalart_allmaras(dbl time_value, /* current time */
                                 dbl tt,         /* parameter to vary time integration from
                                                    explicit (tt = 1) to implicit (tt = 0)    */
                                 dbl dt,         /* current time step size                    */
                                 const PG_DATA *pg_data);
int ad_assemble_turb_k_modified(dbl time_value, /* current time */
                                dbl tt,         /* parameter to vary time integration from
                                                   explicit (tt = 1) to implicit (tt = 0)    */
                                dbl dt,         /* current time step size                    */
                                const PG_DATA *pg_data);
int ad_assemble_turb_omega_modified(dbl time_value, /* current time */
                                    dbl tt,         /* parameter to vary time integration from
                                                       explicit (tt = 1) to implicit (tt = 0)    */
                                    dbl dt,         /* current time step size                    */
                                    const PG_DATA *pg_data);
int ad_assemble_turb_k_omega_modified(dbl time_value, /* current time */
                                      dbl tt,         /* parameter to vary time integration from
                                                         explicit (tt = 1) to implicit (tt = 0)    */
                                      dbl dt, /* current time step size                    */
                                      const PG_DATA *pg_data);
int ad_assemble_k_omega_sst_modified(dbl time_value, /* current time */
                                     dbl tt,         /* parameter to vary time integration from
                                                        explicit (tt = 1) to implicit (tt = 0)    */
                                     dbl dt,         /* current time step size                    */
                                     const PG_DATA *pg_data);
int ad_assemble_invariant(double tt,  /* parameter to vary time integration from
                                       * explicit (tt = 1) to implicit (tt = 0)    */
                          double dt); /*  time step size                          */
void ad_omega_wall_func(double func[DIM], double d_func[DIM][MAX_VARIABLE_TYPES + MAX_CONC][MDE]);

#ifdef __cplusplus
}
#endif

#endif

#endif // GOMA_AD_TURBULENCE_H