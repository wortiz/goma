#ifndef GOMA_AD_LUBRICATION_H
#define GOMA_AD_LUBRICATION_H
#ifdef __cplusplus
extern "C" {
#include "exo_struct.h"
}
#include <memory>
#include "ad/structs.h"
struct AD_Lubrication_Auxiliaries {
  ADType q[DIM];             /* Volumetric flow rate per unit width */
  ADType v_avg[DIM];         /* Average velocity, i.e. q divided by height */
  ADType gradP[DIM];         /* Composite pressure gradient vector */
  ADType gradP_mag;          /* Magnitude of pressure gradient */
  ADType gradP_tangent[DIM]; /* Tangent vector of the pressure gradient */
  ADType gradP_normal[DIM];  /* Unit vector perpendicular to the pressure */
  ADType H;                  /* Lubrication Gap Height */
  ADType H_cap;              /* Lubrication Gap Height unaffected by wall effects */
  ADType srate;              /* Lubrication Characteristic Shear Rate */
  ADType mu_star;            /* Lubrication Characteristic Viscosity */
  ADType op_curv;            /* Lubrication Out-of-plane Curvature */
  ADType visc_diss;          /* Lubrication Integrated Viscous Dissipation */

  ADType dgradP_mag_dP;          /* Pressure gradient magnitude sensitivities w.r.t.
                                    pressure */
  ADType dgradP_tangent_dP[DIM]; /* Pressure gradient tangent sensitivities
                                    w.r.t. pressure */
  ADType dgradP_normal_dP[DIM];  /* Pressure gradient normal sensitivities w.r.t.
                                    pressure */

  ADType dq_dh[DIM][MDE];           /* Flow rate sensitivities w.r.t. height */
  ADType dq_dh1[DIM][MDE];          /* Flow rate sensitivities w.r.t. height */
  ADType dq_dh2[DIM][MDE];          /* Flow rate sensitivities w.r.t. height */
  ADType dq_dp[DIM][MDE];           /* Flow rate sensitivities w.r.t. lubrication pressure */
  ADType dq_dp2[DIM];               /* Flow rate sensitivities w.r.t. lubrication pressure */
  ADType dq_df[DIM][MDE];           /* Flow rate sensitivities w.r.t. level set */
  ADType dq_dk[DIM];                /* Flow rate sensitivities w.r.t. curvature */
  ADType dq_dx[DIM][DIM][MDE];      /* Flow rate sensitivities w.r.t. mesh deformation */
  ADType dq_dnormal[DIM][DIM][MDE]; /* Flow rate sensitivities w.r.t. shell normal */
  ADType dq_drs[DIM][DIM][MDE];     /* Flow rate sensitivities w.r.t. real solid
                                       deformation */
  ADType dq_ddh[DIM];               /* Flow rate sensitivities w.r.t. heat transport */
  ADType dq_dc[DIM][MDE];           /* Flow rate sensitivities w.r.t. particles volume
                                       fraction */
  ADType dq_dconc[DIM][DIM][MAX_CONC]
                 [MDE];             /* Flow rate sensitivities w.r.t. species concentration */
  ADType dq_dshear_top[DIM][MDE];   /* Flow rate sensitivities w.r.t. top wall
                                       shear rate */
  ADType dq_dshear_bot[DIM][MDE];   /* Flow rate sensitivities w.r.t. bottom wall
                                       shear rate */
  ADType dq_dcross_shear[DIM][MDE]; /* Flow rate sensitivities w.r.t. cross
                                       stream shear stress */
  ADType dq_dgradp[DIM][DIM];       /* Flow rate sensitivities w.r.t. pressure gradient */
  ADType dq_dT[DIM];                /* Flow rate sensitivities w.r.t. Temperature */
  ADType dq_dshrw[DIM];             /* Flow rate sensitivities w.r.t. Wall shear rate */
  ADType dq_dv[DIM][DIM][MDE];      /* Flow rate sensitivities w.r.t. velocities */

  ADType dv_avg_dh[DIM][MDE];           /* Average velocity sensitivities w.r.t. height */
  ADType dv_avg_dh1[DIM][MDE];          /* Average velocity sensitivities w.r.t. height */
  ADType dv_avg_dh2[DIM][MDE];          /* Average velocity sensitivities w.r.t. height */
  ADType dv_avg_dp2[DIM];               /* Average velocity sensitivities w.r.t.
                                                lubrication pressure */
  ADType dv_avg_dnormal[DIM][DIM][MDE]; /* Average velocity sensitivities w.r.t. mesh deformation */
  ADType dv_avg_df[DIM][MDE];           /* Average velocity sensitivities w.r.t. level set */
  ADType dv_avg_dk[DIM];                /* Average veloctiy sensitivities w.r.t. curvature */
  ADType dv_avg_dx[DIM][DIM][MDE];      /* Average velocity sensitivities w.r.t. mesh
                                           deformation */
  ADType dv_avg_drs[DIM][DIM][MDE];     /* Average velocity sensitivities w.r.t.
                                           real solid deformation*/
  ADType dv_avg_ddh[DIM];               /* Average velocity sensitivities w.r.t. heat
                                                transport */
  ADType dv_avg_dc[DIM][MDE];           /* Average velocity sensitivities w.r.t. particles
                                           volume fraction */
  ADType dv_avg_dconc[DIM][DIM][MAX_CONC]
                     [MDE]; /* Average velocity sensitivities w.r.t. species concentration */
  ADType dv_avg_dshear_top[DIM][MDE];   /* Average velocity sensitivities w.r.t.
                                           top wall shear rate */
  ADType dv_avg_dshear_bot[DIM][MDE];   /* Average velocity sensitivities w.r.t.
                                           bottom wall shear rate */
  ADType dv_avg_dcross_shear[DIM][MDE]; /* Average velocity sensitivities w.r.t.
                                           cross stream shear stress */
  ADType dv_dgradp[DIM][DIM];     /* Average velocity sensitivities w.r.t. pressure gradient */
  ADType dv_avg_dT[DIM];          /* Average velosity sensitivities w.r.t. Temperature */
  ADType dv_avg_dshrw[DIM];       /* Average velosity sensitivities w.r.t. Wall shear rate */
  ADType dH_dmesh[DIM][MDE];      /* lubrication gap sensitivities w.r.t. mesh */
  ADType dH_drealsolid[DIM][MDE]; /* lubrication gap sensitivities w.r.t. real
                                     solid */
  ADType dH_dp;                   /* lubrication gap sensitivities w.r.t. pressure */
  ADType dH_ddh;                  /* lubrication gap sensitivities w.r.t. added height */
  ADType dop_curv_dx[DIM][MDE];   /* Out-of-plane Curvature sensitivities w.r.t. mesh deformation */
  ADType dop_curv_df[MDE];        /* Out-of-plane Curvature sensitivities w.r.t. level set */
  ADType dvisc_diss_dT; /* Lubrication Integrated Viscous Dissipation Sensitivity to Temperature */
  ADType dvisc_diss_dpgrad; /* Lubrication Integrated Viscous Dissipation Sensitivity to Pgrad */
};

extern std::unique_ptr<AD_Lubrication_Auxiliaries> AD_LubAux;

void ad_calculate_lub_q_v(const int EQN, double time, double dt, double xi[DIM], const Exo_DB *exo);

void ADInn(ADType v[DIM], // Input vector
           ADType w[DIM]  // Output rotated vector
);

ADType ad_height_function_model(ADType *H_U,
                             ADType *H_L,
                             ADType *dH_U_dtime,
                             ADType *dH_L_dtime,
                             ADType dH_U_dX[DIM],
                             ADType dH_L_dX[DIM],
                             ADType *dH_U_dp,
                             ADType *dH_U_ddh,
                             ADType dH_dF[MDE],
                             double time,    /* present time value           */
                             double delta_t); /* present time step             */
extern "C" {
#endif
#include "el_elm.h"
#include "exo_struct.h"
#include "mm_as_structs.h"
int ad_assemble_lubrication(const int EQN,  /* equation type: either R_LUBP or R_LUBP2 */
                            double time,    /* present time value */
                            double tt,      /* parameter to vary time integration from
                                             * explicit (tt = 1) to implicit (tt = 0)    */
                            double dt,      /* current time step size */
                            double xi[DIM], /* Local stu coordinates */
                            const Exo_DB *exo);

int ad_assemble_lubrication_curvature(double time,            /* present time value */
                                   double tt,              /* parameter to vary time integration  */
                                   double dt,              /* current time step size */
                                   const PG_DATA *pg_data, /* Element scales */
                                   double xi[DIM],         /* Local stu coordinates */
                                   const Exo_DB *exo);    /* Exodus database */
#ifdef __cplusplus
}
#endif

#endif // GOMA_AD_LUBRICATION_H