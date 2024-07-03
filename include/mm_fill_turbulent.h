#ifndef GOMA_MM_FILL_TURBULENT_H
#define GOMA_MM_FILL_TURBULENT_H

#ifdef EXTERN
#undef EXTERN
#endif

#ifdef GOMA_MM_FILL_TURBULENT_C
#define EXTERN
#else
#define EXTERN extern
#endif

#include "exo_struct.h"
#include "mm_as_structs.h"
#include "std.h"

struct turb_dependence {
  double d_diss[MDE];
  double d_k[MDE];
  double d_v[DIM][MDE];
  double d_mesh[DIM][MDE];
};

struct k_omega_sst_const {
  double gamma1;
  double gamma2;
  double sigma_k1;
  double sigma_k2;
  double sigma_omega1;
  double sigma_omega2;
  double beta1;
  double beta2;
  double beta_star;
  double kappa;
  double a1;
  double CD_komega_min;
  double Plim_factor;
};

EXTERN int assemble_spalart_allmaras(dbl time_value, /* current time */
                                     dbl tt,         /* parameter to vary time integration from
                                                      explicit (tt = 1) to implicit (tt = 0)    */
                                     dbl dt,         /* current time step size                    */
                                     const PG_DATA *pg_data);

int assemble_turbulent_kinetic_energy_sst(dbl time_value, /* current time */
                                          dbl tt,         /* parameter to vary time integration from
                                                             explicit (tt = 1) to implicit (tt = 0)    */
                                          dbl dt, /* current time step size                    */
                                          const PG_DATA *pg_data);

int assemble_turbulent_dissipation_sst(dbl time_value, /* current time */
                                          dbl tt,         /* parameter to vary time integration from
                                                             explicit (tt = 1) to implicit (tt = 0)    */
                                          dbl dt, /* current time step size                    */
                                          const PG_DATA *pg_data);


void set_k_omega_sst_const_2003(struct k_omega_sst_const *k_omega_sst);

void sst_viscosity(const struct k_omega_sst_const *constants,
                   dbl *mu_turb,
                   struct turb_dependence *d_mu_turb);
                   
#endif

