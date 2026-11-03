/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) 2023 Goma Developers, National Technology & Engineering   *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

#include "density.h"
#include "ac_particles.h"
#ifdef GOMA_ENABLE_AZTEC
#include "az_aztec.h"
#endif
#include "bc_colloc.h"
#include "bc_contact.h"
#include "el_elm.h"
#include "el_elm_info.h"
#include "el_geom.h"
#include "exo_struct.h"
#include "load_field_variables.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "mm_dil_viscosity.h"
#include "mm_eh.h"
#include "mm_fill_aux.h"
#include "mm_fill_common.h"
#include "mm_fill_fill.h"
#include "mm_fill_ls.h"
#include "mm_fill_momentum.h"
#include "mm_fill_population.h"
#include "mm_fill_ptrs.h"
#include "mm_fill_rs.h"
#include "mm_fill_shell.h"
#include "mm_fill_solid.h"
#include "mm_fill_species.h"
#include "mm_fill_stabilization.h"
#include "mm_fill_stress.h"
#include "mm_fill_terms.h"
#include "mm_fill_util.h"
#include "mm_flux.h"
#include "mm_mp.h"
#include "mm_mp_const.h"
#include "mm_mp_structs.h"
#include "mm_ns_bc.h"
#include "mm_post_def.h"
#include "mm_qtensor_model.h"
#include "mm_shell_util.h"
#include "mm_species.h"
#include "mm_unknown_map.h"
#include "mm_viscosity.h"
#include "rf_allo.h"
#include "rf_bc.h"
#include "rf_bc_const.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_solver.h"
#include "rf_vars_const.h"
#include "sl_util.h"
#include "sl_util_structs.h"
#include "std.h"
#include "stdbool.h"
#include "user_mp.h"
#include "user_mp_gen.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/********************************************************************************/

double density(DENSITY_DEPENDENCE_STRUCT *d_rho, double time)

/**************************************************************************
 *
 * density
 *
 *   Calculate the density and its derivatives at the local gauss point
 *
 * Output
 * -----
 *    dbl d_rho->T[j]    -> derivative of density wrt the jth
 *                          Temperature unknown in an element
 *    dbl d_rho->C[k][j] -> derivative of density wrt the jth
 *                          species unknown of ktype, k, in the element.
 *    dbl d_rho->F[j]    -> derivative of density wrt the jth
 *                          FILL unknown in an element
 *
 *  Return
 * --------
 *    Actual value of the density
 ***************************************************************************/
{
  int w, j, var, var_offset, matrl_species_var_type, dropped_last_species_eqn;
  int species, err, imtrx;
  dbl vol = 0, rho = 0, rho_f, rho_s, pressureThermo, RGAS_CONST;
  dbl avgMolecWeight = 0, tmp;
  double *phi_ptr;

  dbl sv[MAX_CONC];
  dbl sv_p, sum_sv = 0;
  struct Level_Set_Data *ls_old;

  /*
   * Get the material's species variable type
   *    -> We will assume here that all species have the same var type.
   */
  matrl_species_var_type = goma_ctx->fem.mp->Species_Var_Type;

  /*
   * Decide whether the continuity equation for the last species
   * in the species list has been dropped in favor of an implicit
   * sum of mass fractions equals one constraint. If it has, then
   * that implicit constraint has to be reflected in the Jacobian terms
   * calculated from this routine. Basically, any change in the
   * mass fraction or concentration of one species is compensated by a
   * change in the mass fraction or concentration of the last species
   * in the mechanism.
   */
  dropped_last_species_eqn = goma_ctx->fem.mp->Dropped_Last_Species_Eqn;

  /* Zero out sensitivities */
  if (d_rho != NULL) {
    zeroStructures(d_rho, 1);
  }

  /*
   *   Branch according to the density model from the material properties
   *   structure
   */
  if (goma_ctx->fem.mp->DensityModel == CONSTANT) {
    rho = goma_ctx->fem.mp->density;
  } else if (goma_ctx->fem.mp->DensityModel == USER) {
    (void)usr_density(goma_ctx->fem.mp->u_density);
    rho = goma_ctx->fem.mp->density;

    if (d_rho != NULL) {
      var = TEMPERATURE;
      for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
        d_rho->T[j] = goma_ctx->fem.mp->d_density[var] * goma_ctx->fem.bf[var]->phi[j];
      }

      if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][MASS_FRACTION]) {
        for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
          var = MASS_FRACTION;
          var_offset = MAX_VARIABLE_TYPES + w;
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            d_rho->C[w][j] =
                goma_ctx->fem.mp->d_density[var_offset] * goma_ctx->fem.bf[var]->phi[j];
          }
        }
      }
    }

  } else if (goma_ctx->fem.mp->DensityModel == DENSITY_THERMEXP) {
    rho = goma_ctx->fem.mp->u_density[0] /
          (1. + goma_ctx->fem.mp->u_density[1] *
                    (goma_ctx->fem.fv->T - goma_ctx->fem.mp->reference[TEMPERATURE]));
    if (d_rho != NULL) {
      var = TEMPERATURE;
      for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
        d_rho->T[j] = -rho * goma_ctx->fem.mp->u_density[1] /
                      (1. + goma_ctx->fem.mp->u_density[1] *
                                (goma_ctx->fem.fv->T - goma_ctx->fem.mp->reference[TEMPERATURE])) *
                      goma_ctx->fem.bf[var]->phi[j];
      }
    }

  } else if (goma_ctx->fem.mp->DensityModel == FILL) {

    rho = goma_ctx->fem.mp->u_density[0];

    if (goma_ctx->fem.fv->F < 0.1) {
      rho = goma_ctx->fem.mp->u_density[1];
    }

  } else if (goma_ctx->fem.mp->DensityModel == DENSITY_LEVEL_SET) {
    double rho0 = goma_ctx->fem.mp->u_density[0];
    double rho1 = goma_ctx->fem.mp->u_density[1];
    double width = goma_ctx->fem.mp->u_density[2];
    if (d_rho == NULL)
      err = level_set_property(rho0, rho1, width, &rho, NULL);
    else
      err = level_set_property(rho0, rho1, width, &rho, d_rho->F);
    GOMA_EH(err, "level_set_property() failed for density.");
  }

  else if (goma_ctx->fem.mp->DensityModel == DENSITY_CONST_PHASE_FUNC) {
    int num, a;
    double width;
    double rho1, rho2 = 0, tmp_rho;
    struct Level_Set_Data *ls_save = goma_ctx->fem.ls;

    num = goma_ctx->fem.pfd->num_phase_funcs;
    width = goma_ctx->fem.mp->u_density[num];
    // Major cludgey here.  this is fubar'd like viscosity

    for (a = 0; a < num; a++) {
      rho1 = goma_ctx->fem.mp->u_density[a];
      rho1 = goma_ctx->fem.mp->u_density[a + 2];
      goma_ctx->fem.ls = goma_ctx->fem.pfd->ls[a];

      if (d_rho != NULL)
        err = level_set_property(rho1, rho2, width, &tmp_rho, d_rho->pf[a]);
      else
        err = level_set_property(rho1, rho2, width, &tmp_rho, NULL);

      rho += tmp_rho;
    }
    if (fabs(rho) < DBL_SMALL)
      rho = goma_ctx->fem.mp->u_density[num + 1];
    goma_ctx->fem.ls = ls_save;

  }

  else if (goma_ctx->fem.mp->DensityModel == DENSITY_FOAM) {
    double x0, Rgas, MW, rho_epoxy, rho_fluor, T, vol, Press;
    species = (int)goma_ctx->fem.mp->u_density[0]; /* species number fluorinert */
    x0 = goma_ctx->fem.mp->u_density[1];           /* Initial fluorinert mass fraction */
    Rgas = goma_ctx->fem.mp->u_density[2];         /* Gas constant in appropriate units */
    MW = goma_ctx->fem.mp->u_density[3];
    rho_epoxy = goma_ctx->fem.mp->u_density[4]; /* Density of epoxy resin */
    rho_fluor = goma_ctx->fem.mp->u_density[5]; /* Density of liquid fluorinert */

    if (goma_ctx->fem.fv->c[species] > 0.)
      vol = goma_ctx->fem.fv->c[species];
    else
      vol = 0.;

    if (vol > x0)
      vol = x0;

    T = goma_ctx->fem.fv->T;
    Press = goma_ctx->fem.upd->Pressure_Datum;
    rho = 1. / ((x0 - vol) * Rgas * T / (Press * MW) + (1.0 - x0) / rho_epoxy + vol / rho_fluor);

    var = MASS_FRACTION;
    if (vol > 0. && d_rho != NULL) {
      if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
        if ((vol > 0.) && (vol < x0))
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            d_rho->C[species][j] = rho * rho * (Rgas * T / (Press * MW) - 1. / rho_fluor) *
                                   goma_ctx->fem.bf[var]->phi[j];
          }
      }
    }

    var = TEMPERATURE;
    if (d_rho != NULL) {
      if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
        for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
          d_rho->T[j] =
              -rho * rho * ((x0 - vol) * Rgas / (Press * MW)) * goma_ctx->fem.bf[var]->phi[j];
        }
      }
    }

  } else if (goma_ctx->fem.mp->DensityModel == DENSITY_FOAM_PBE) {
    int species_BA_g;
    int species_CO2_g;
    int species_CO2_l;
    int species_BA_l;
    int err;
    err = get_foam_pbe_indices(NULL, NULL, &species_BA_l, &species_BA_g, &species_CO2_l,
                               &species_CO2_g);
    if (err)
      return 0;

    double M_BA = goma_ctx->fem.mp->u_species_source[species_BA_l][0];
    double M_CO2 = goma_ctx->fem.mp->u_species_source[species_CO2_l][0];
    double rho_bubble = 0;
    double rho_foam = goma_ctx->fem.mp->u_density[0];
    double ref_press = goma_ctx->fem.mp->u_density[1];
    double Rgas_const = goma_ctx->fem.mp->u_density[2];

    if (goma_ctx->fem.fv->c[species_BA_g] > PBE_FP_SMALL ||
        goma_ctx->fem.fv->c[species_CO2_g] > PBE_FP_SMALL) {
      rho_bubble =
          (ref_press / (Rgas_const * goma_ctx->fem.fv->T)) *
          (goma_ctx->fem.fv->c[species_CO2_g] * M_CO2 + goma_ctx->fem.fv->c[species_BA_g] * M_BA) /
          (goma_ctx->fem.fv->c[species_CO2_g] + goma_ctx->fem.fv->c[species_BA_g]);
    }

    double inv_mom_frac = 1 / (1 + goma_ctx->fem.fv->moment[1]);
    rho = rho_bubble * (goma_ctx->fem.fv->moment[1] * inv_mom_frac) + rho_foam * inv_mom_frac;

    if (d_rho != NULL) {
      var = TEMPERATURE;
      if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
        for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
          d_rho->T[j] = (-rho_bubble) / goma_ctx->fem.fv->T *
                        (goma_ctx->fem.fv->moment[1] * inv_mom_frac) *
                        goma_ctx->fem.bf[var]->phi[j];
        }
      }

      var = MOMENT1;
      if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
        for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
          d_rho->moment[1][j] =
              (rho_bubble * inv_mom_frac * inv_mom_frac - rho_foam * inv_mom_frac * inv_mom_frac) *
              goma_ctx->fem.bf[var]->phi[j];
        }
      }

      var = MASS_FRACTION;
      if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
        for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
          if (goma_ctx->fem.fv->c[species_BA_g] > PBE_FP_SMALL ||
              goma_ctx->fem.fv->c[species_CO2_g] > PBE_FP_SMALL) {
            d_rho->C[species_BA_g][j] =
                (goma_ctx->fem.fv->moment[1] * inv_mom_frac) * goma_ctx->fem.bf[var]->phi[j] *
                (ref_press / (Rgas_const * goma_ctx->fem.fv->T)) *
                ((M_BA - M_CO2) * goma_ctx->fem.fv->c[species_CO2_g]) /
                ((goma_ctx->fem.fv->c[species_CO2_g] + goma_ctx->fem.fv->c[species_BA_g]) *
                 (goma_ctx->fem.fv->c[species_CO2_g] + goma_ctx->fem.fv->c[species_BA_g]));

            d_rho->C[species_CO2_g][j] =
                (goma_ctx->fem.fv->moment[1] * inv_mom_frac) * goma_ctx->fem.bf[var]->phi[j] *
                (ref_press / (Rgas_const * goma_ctx->fem.fv->T)) *
                ((M_CO2 - M_BA) * goma_ctx->fem.fv->c[species_BA_g]) /
                ((goma_ctx->fem.fv->c[species_CO2_g] + goma_ctx->fem.fv->c[species_BA_g]) *
                 (goma_ctx->fem.fv->c[species_CO2_g] + goma_ctx->fem.fv->c[species_BA_g]));
          }
        }
      }
    }
  } else if (goma_ctx->fem.mp->DensityModel == DENSITY_FOAM_PBE_EQN) {
    rho = goma_ctx->fem.fv->rho;

    var = DENSITY_EQN;
    if (d_rho != NULL) {
      for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
        d_rho->rho[j] = goma_ctx->fem.bf[var]->phi[j];
      }
    }
  } else if (goma_ctx->fem.mp->DensityModel == DENSITY_FOAM_CONC) {
    double Rgas, MW_f, MW_a, rho_epoxy, rho_fluor, T, Press;
    dbl rho_v_inv, d_rho_v_inv_dT, rho_a_inv, d_rho_a_inv_dT, drho_T, drho_c_v, drho_c_a, drho_c_l;
    int species_l, species_v, species_a;

    species_l = (int)goma_ctx->fem.mp->u_density[0]; /* species number fluorinert liquid */
    species_v = (int)goma_ctx->fem.mp->u_density[1]; /* species number fluorinert vapor */
    species_a = (int)goma_ctx->fem.mp->u_density[2]; /* species number air vapor */
    Rgas = goma_ctx->fem.mp->u_density[3];           /* Gas constant in appropriate units */
    MW_f = goma_ctx->fem.mp->u_density[4];           /* molecular weight fluorinert */
    MW_a = goma_ctx->fem.mp->u_density[5];           /* molecular weight air */
    rho_epoxy = goma_ctx->fem.mp->u_density[6];      /* Density of epoxy resin */
    rho_fluor = goma_ctx->fem.mp->u_density[7];      /* Density of liquid fluorinert */

    T = goma_ctx->fem.fv->T;
    Press = goma_ctx->fem.upd->Pressure_Datum;
    rho_v_inv = Rgas * T / (Press * MW_f);
    d_rho_v_inv_dT = Rgas / (Press * MW_f);
    rho_a_inv = Rgas * T / (Press * MW_a);
    d_rho_a_inv_dT = Rgas / (Press * MW_a);

    drho_c_v = 1. - rho_epoxy * rho_v_inv;
    drho_c_a = 1. - rho_epoxy * rho_a_inv;
    drho_c_l = 1. - rho_epoxy / rho_fluor;

    drho_T = -goma_ctx->fem.fv->c[species_v] * rho_epoxy * d_rho_v_inv_dT -
             goma_ctx->fem.fv->c[species_a] * rho_epoxy * d_rho_a_inv_dT;

    var = MASS_FRACTION;
    if (d_rho != NULL) {
      if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
        for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
          d_rho->C[species_v][j] = drho_c_v * goma_ctx->fem.bf[var]->phi[j];
          d_rho->C[species_a][j] = drho_c_a * goma_ctx->fem.bf[var]->phi[j];
          d_rho->C[species_l][j] = drho_c_l * goma_ctx->fem.bf[var]->phi[j];
        }
      }

      var = TEMPERATURE;
      if (d_rho != NULL) {
        if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            d_rho->T[j] = drho_T * goma_ctx->fem.bf[var]->phi[j];
          }
        }
      }
    }
  } else if (goma_ctx->fem.mp->DensityModel == DENSITY_FOAM_TIME) {
    double rho_init, rho_final, aexp, time_delay, realtime;
    rho_init = goma_ctx->fem.mp->u_density[0];   /* Initial density */
    rho_final = goma_ctx->fem.mp->u_density[1];  /* final density */
    aexp = goma_ctx->fem.mp->u_density[2];       /* Arhennius constant for time exponent*/
    time_delay = goma_ctx->fem.mp->u_density[3]; /* time delay before foaming starts */

    if (time > time_delay) {
      realtime = time - time_delay;
      rho = rho_final + (rho_init - rho_final) * exp(-aexp * realtime);
    } else {
      rho = rho_init;
    }
  } else if (goma_ctx->fem.mp->DensityModel == DENSITY_FOAM_TIME_TEMP) {
    double T;
    double rho_init = goma_ctx->fem.mp->u_density[0];   /* Initial density (units of gm cm-3 */
    double rho_final = goma_ctx->fem.mp->u_density[1];  /* Final density (units of gm cm-3 */
    double cexp = goma_ctx->fem.mp->u_density[2];       /* cexp in units of Kelvin sec */
    double coffset = goma_ctx->fem.mp->u_density[3];    /* offset in units of seconds */
    double time_delay = goma_ctx->fem.mp->u_density[4]; /* time delay in units of seconds */
    double drhoDT;
    T = goma_ctx->fem.fv->T;
    if (time > time_delay) {
      double realtime = time - time_delay;
      double delRho = (rho_init - rho_final);
      double cdenom = cexp - coffset * T;
      double expT = exp(-realtime * T / cdenom);
      rho = rho_final + delRho * expT;
      drhoDT = delRho * expT * (-realtime * cexp / (cdenom * cdenom));
    } else {
      rho = rho_init;
      drhoDT = 0.0;
    }
    var = TEMPERATURE;
    if (d_rho != NULL) {
      if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
        for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
          d_rho->T[j] = drhoDT * goma_ctx->fem.bf[var]->phi[j];
        }
      }
    }
  } else if (goma_ctx->fem.mp->DensityModel == DENSITY_FOAM_PMDI_10) {
    int var, j;
    int w;
    double volF = goma_ctx->fem.mp->volumeFractionGas;

    double M_CO2 = goma_ctx->fem.mp->u_density[0];
    double rho_liq = goma_ctx->fem.mp->u_density[1];
    double ref_press = goma_ctx->fem.mp->u_density[2];
    double Rgas_const = goma_ctx->fem.mp->u_density[3];

    double rho_gas = 0;

    if (goma_ctx->fem.fv->T > 0) {
      rho_gas = (ref_press * M_CO2 / (Rgas_const * goma_ctx->fem.fv->T));
    }

    rho = rho_gas * volF + rho_liq * (1 - volF);

    /* Now do sensitivies */

    var = MASS_FRACTION;
    if (vol > 0. && d_rho != NULL) {
      if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
        for (w = 0; w < goma_ctx->fem.pd->Num_Species; w++) {
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            d_rho->C[w][j] =
                (rho_gas * goma_ctx->fem.mp->d_volumeFractionGas[MAX_VARIABLE_TYPES + w] -
                 rho_liq * goma_ctx->fem.mp->d_volumeFractionGas[MAX_VARIABLE_TYPES + w]) *
                goma_ctx->fem.bf[var]->phi[j];
          }
        }
      }
    }

    var = TEMPERATURE;
    if (d_rho != NULL) {
      if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
        for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
          if (goma_ctx->fem.fv->T > 0) {
            d_rho->T[j] = (-rho_gas / goma_ctx->fem.fv->T * volF +
                           rho_gas * goma_ctx->fem.mp->d_volumeFractionGas[var] -
                           rho_liq * goma_ctx->fem.mp->d_volumeFractionGas[var]) *
                          goma_ctx->fem.bf[var]->phi[j];
          } else {
            d_rho->T[j] = (rho_gas * goma_ctx->fem.mp->d_volumeFractionGas[var] -
                           rho_liq * goma_ctx->fem.mp->d_volumeFractionGas[var]) *
                          goma_ctx->fem.bf[var]->phi[j];
          }
        }
      }
    }

  }

  else if (goma_ctx->fem.mp->DensityModel == DENSITY_MOMENT_BASED) {
    int var, j;
    int w;
    double volF = goma_ctx->fem.mp->volumeFractionGas;

    double rho_gas = goma_ctx->fem.mp->u_density[0];
    double rho_liq = goma_ctx->fem.mp->u_density[1];

    rho = rho_gas * volF + rho_liq * (1 - volF);

    /* Now do sensitivies */

    var = MASS_FRACTION;
    if (vol > 0. && d_rho != NULL) {
      if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
        for (w = 0; w < goma_ctx->fem.pd->Num_Species; w++) {
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            d_rho->C[w][j] = 0;
          }
        }
      }
    }

    var = TEMPERATURE;
    if (d_rho != NULL) {
      if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
        for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
          if (goma_ctx->fem.fv->T > 0) {
            d_rho->T[j] = 0;
          } else {
            d_rho->T[j] = 0;
          }
        }
      }
    }

  } else if (goma_ctx->fem.mp->DensityModel == SUSPENSION) {
    species = (int)goma_ctx->fem.mp->u_density[0];
    rho_f = goma_ctx->fem.mp->u_density[1];
    rho_s = goma_ctx->fem.mp->u_density[2];

    vol = goma_ctx->fem.fv->c[species];
    if (vol < 0.) {
      vol = 0.;
    }

    rho = rho_f + (rho_s - rho_f) * vol;
    var = MASS_FRACTION;
    if (vol > 0. && d_rho != NULL) {
      if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
        for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
          d_rho->C[species][j] = (rho_s - rho_f) * goma_ctx->fem.bf[var]->phi[j];
        }
      }
    }
  } else if (goma_ctx->fem.mp->DensityModel == SUSPENSION_PM) {
    /* For the SUSPENSION_PM model, the density is "usually" just the
     * fluid density.  It is only in two cases that the density we want is
     * actually the particle phase density.  1) In the particle phase
     * momentum equations. 2) In the particle phase advection equation
     * (like a chemical species).  In those two cases, this muse be "undone".
     * This seemed the best way to do it so there was a reasonable default
     * behavior, and you didn't need a bunch of if()'s bracketing every
     * density() call...
     */

    rho = goma_ctx->fem.mp->u_density[1];

    /* This doesn't account for other species present when calculating
     * density.  But they are currently just "collected" into one rho_f,
     * so all of the d_rho->C's should be zero!.
     */
    /*
      var = MASS_FRACTION;
      if(vol > 0.)
      {
      if (pd->v[pg->imtrx][var] )
      {
      for ( j=0; j<ei[pg->imtrx]->dof[var]; j++)
      {
      d_rho->C[species][j] = (rho_s - rho_f)*bf[var]->phi[j];
      }
      }
      }
    */
  } else if (goma_ctx->fem.mp->DensityModel == THERMAL_BATTERY) {
    rho = goma_ctx->fem.mp->u_density[0] -
          goma_ctx->fem.mp->u_density[1] * (2.0 * goma_ctx->fem.fv->c[0]);
    /* Ref.: Pollard & Newman 1981, p.501 */

    if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][MASS_FRACTION] && d_rho != NULL) {
      var = MASS_FRACTION;
      for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
        d_rho->C[0][j] = -goma_ctx->fem.mp->u_density[1] * 2.0 * goma_ctx->fem.bf[var]->phi[j];
      }
    }

  } else if (goma_ctx->fem.mp->DensityModel == DENSITY_IDEAL_GAS) {
    /*
     * Specify the thermodynamic pressure here
     *   -> For the moment we ignore any coupling between
     *      the dynamic pressure field and the thermodynamic
     *      pressure field.
     */
    pressureThermo = goma_ctx->fem.upd->Pressure_Datum;

    /*
     * Specify the gas constant according to CRC IUPAC conventions
     */
    RGAS_CONST = 8.314510E7; /*    g cm^2/(sec^2 g-mole K)  */

    /*
     *  Base density calculation depends on species var type
     */
    switch (matrl_species_var_type) {
    case SPECIES_MASS_FRACTION:
      avgMolecWeight = wt_from_Yk(goma_ctx->fem.mp->Num_Species, goma_ctx->fem.fv->c,
                                  goma_ctx->fem.mp->molecular_weight);
      rho = pressureThermo * avgMolecWeight / (RGAS_CONST * goma_ctx->fem.fv->T);
      break;
    case SPECIES_MOLE_FRACTION:
      avgMolecWeight = wt_from_Xk(goma_ctx->fem.mp->Num_Species, goma_ctx->fem.fv->c,
                                  goma_ctx->fem.mp->molecular_weight);
      rho = pressureThermo * avgMolecWeight / (RGAS_CONST * goma_ctx->fem.fv->T);
      break;
    case SPECIES_CONCENTRATION:
      rho = 0.0;
      for (w = 0; w < goma_ctx->fem.mp->Num_Species; w++) {
        rho += goma_ctx->fem.fv->c[w] * goma_ctx->fem.mp->molecular_weight[w];
      }

      break;
    default:
      fprintf(stderr, "Density error: species var type not handled: %d\n", matrl_species_var_type);
      exit(-1);
    }

    /*
     * Now do the Jacobian terms
     *          HKM -> No dependence on the pressure is calculated
     *
     * Temperature dependence is common to all species var types
     */
    if (d_rho != NULL && goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][TEMPERATURE]) {
      phi_ptr = goma_ctx->fem.bf[MASS_FRACTION]->phi;
      tmp = -rho / goma_ctx->fem.fv->T;
      for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[TEMPERATURE]; j++) {
        d_rho->T[j] = tmp * phi_ptr[j];
      }
    }
    /*
     * Dependence on the species unknowns.
     */
    if (d_rho != NULL && goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][MASS_FRACTION]) {
      phi_ptr = goma_ctx->fem.bf[MASS_FRACTION]->phi;
      switch (matrl_species_var_type) {
      case SPECIES_MASS_FRACTION:
        if (dropped_last_species_eqn) {
          for (w = 0; w < goma_ctx->fem.mp->Num_Species_Eqn; w++) {
            tmp = -rho * avgMolecWeight *
                  (1.0 / (goma_ctx->fem.mp->molecular_weight[w]) -
                   1.0 / (goma_ctx->fem.mp->molecular_weight[goma_ctx->fem.mp->Num_Species_Eqn]));
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[MASS_FRACTION]; j++) {
              d_rho->C[w][j] = tmp * phi_ptr[j];
            }
          }
        } else {
          for (w = 0; w < goma_ctx->fem.mp->Num_Species_Eqn; w++) {
            tmp = -rho * avgMolecWeight / (goma_ctx->fem.mp->molecular_weight[w]);
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[MASS_FRACTION]; j++) {
              d_rho->C[w][j] = tmp * phi_ptr[j];
            }
          }
        }
        break;
      case SPECIES_MOLE_FRACTION:
        if (dropped_last_species_eqn) {
          for (w = 0; w < goma_ctx->fem.mp->Num_Species_Eqn; w++) {
            tmp = rho / avgMolecWeight *
                  (goma_ctx->fem.mp->molecular_weight[w] -
                   goma_ctx->fem.mp->molecular_weight[goma_ctx->fem.mp->Num_Species_Eqn]);
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[MASS_FRACTION]; j++) {
              d_rho->C[w][j] = tmp * phi_ptr[j];
            }
          }
        } else {
          for (w = 0; w < goma_ctx->fem.mp->Num_Species_Eqn; w++) {
            tmp = rho / avgMolecWeight * goma_ctx->fem.mp->molecular_weight[w];
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[MASS_FRACTION]; j++) {
              d_rho->C[w][j] = tmp * phi_ptr[j];
            }
          }
        }
        break;
      case SPECIES_CONCENTRATION:
        if (dropped_last_species_eqn) {
          for (w = 0; w < goma_ctx->fem.mp->Num_Species_Eqn; w++) {
            tmp = goma_ctx->fem.mp->molecular_weight[w] -
                  goma_ctx->fem.mp->molecular_weight[goma_ctx->fem.mp->Num_Species_Eqn];
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[MASS_FRACTION]; j++) {
              d_rho->C[w][j] = tmp * phi_ptr[j];
            }
          }
        } else {
          for (w = 0; w < goma_ctx->fem.mp->Num_Species_Eqn; w++) {
            tmp = goma_ctx->fem.mp->molecular_weight[w];
            for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[MASS_FRACTION]; j++) {
              d_rho->C[w][j] = tmp * phi_ptr[j];
            }
          }
        }
        break;
      default:
        fprintf(stderr, "Density error: species var type not handled: %d\n",
                matrl_species_var_type);
        exit(-1);
      }
    }
  } else if (goma_ctx->fem.mp->DensityModel == REACTIVE_FOAM) {
    /* added ACSun 04/02 */
    if (goma_ctx->fem.mp->SpeciesSourceModel[0] == FOAM) {
      foam_species_source(goma_ctx->fem.mp->u_species_source[0]);
    } else {
      GOMA_EH(GOMA_ERROR, "Must specify FOAM species source in the material's file");
    }

    rho = 0.;
    for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
      sv[w] = goma_ctx->fem.mp->specific_volume[w];
    }
    sv_p = goma_ctx->fem.mp->specific_volume[goma_ctx->fem.pd->Num_Species_Eqn];

    for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
      sum_sv += (sv[w] - sv_p) * goma_ctx->fem.fv->c[w];
    }
    sum_sv += sv_p;
    rho = 1 / sum_sv;

    var = MASS_FRACTION;
    if (d_rho != NULL && goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
      phi_ptr = goma_ctx->fem.bf[var]->phi;
      for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
        for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
          d_rho->C[w][j] = -rho * rho * (sv[w] - sv_p) * phi_ptr[j];
        }
      }
    }
    var = TEMPERATURE;
    if (d_rho != NULL && goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
      phi_ptr = goma_ctx->fem.bf[var]->phi;
      for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
        d_rho->T[j] = 0.;
        for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
          d_rho->T[j] -= rho * rho * (sv[w] - sv_p) * goma_ctx->fem.mp->species_source[w] * 3.;
        }
        d_rho->T[j] *= phi_ptr[j];
      }
    }
  } else if (goma_ctx->fem.mp->DensityModel == SOLVENT_POLYMER) {
    double *param = goma_ctx->fem.mp->u_density;

    /* added ACSun 7/99 */
    rho = 0.;
    for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
      sv[w] = goma_ctx->fem.mp->specific_volume[w];
    }
    sv_p = param[0];
    var = MASS_FRACTION;

    switch (matrl_species_var_type) {
    case SPECIES_MASS_FRACTION:
      for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
        sum_sv += (sv[w] - sv_p) * goma_ctx->fem.fv->c[w];
      }
      sum_sv += sv_p;
      rho = 1 / sum_sv;

      if (d_rho != NULL && goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][MASS_FRACTION]) {
        phi_ptr = goma_ctx->fem.bf[MASS_FRACTION]->phi;
        for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
          for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
            d_rho->C[w][j] = -rho * rho * (sv[w] - sv_p) * phi_ptr[j];
          }
        }
      }
      break;
    case SPECIES_DENSITY:
      for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
        rho += (1. - sv[w] / sv_p) * goma_ctx->fem.fv->c[w];
      }
      rho += 1. / sv_p;

      if (d_rho != NULL && goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][MASS_FRACTION]) {
        phi_ptr = goma_ctx->fem.bf[MASS_FRACTION]->phi;
        for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
          for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
            d_rho->C[w][j] = (1. - sv[w] / sv_p) * phi_ptr[j];
          }
        }
      }
      break;
    default:
      for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
        sum_sv += (sv[w] - sv_p) * goma_ctx->fem.fv->c[w];
      }
      sum_sv += sv_p;
      rho = 1 / sum_sv;

      if (d_rho != NULL && goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][MASS_FRACTION]) {
        phi_ptr = goma_ctx->fem.bf[MASS_FRACTION]->phi;
        for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
          for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
            d_rho->C[w][j] = -rho * rho * (sv[w] - sv_p) * phi_ptr[j];
          }
        }
      }
      break;
    }
  } else if (goma_ctx->fem.mp->DensityModel == DENSITY_CONSTANT_LAST_CONC) {
    if (matrl_species_var_type != SPECIES_CONCENTRATION) {
      GOMA_EH(GOMA_ERROR, "unimplemented");
    }
    if (goma_ctx->fem.pd->Num_Species > goma_ctx->fem.pd->Num_Species_Eqn) {
      w = goma_ctx->fem.pd->Num_Species_Eqn;
      rho = goma_ctx->fem.mp->molecular_weight[w] * goma_ctx->fem.fv->c[w];
    } else {
      rho = 0.0;
    }
    for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
      rho += goma_ctx->fem.mp->molecular_weight[w] * goma_ctx->fem.fv->c[w];
    }
    if (d_rho) {
      phi_ptr = goma_ctx->fem.bf[MASS_FRACTION]->phi;
      for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
        for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[MASS_FRACTION]; j++) {
          d_rho->C[w][j] = goma_ctx->fem.mp->molecular_weight[w] * phi_ptr[j];
        }
      }
    }
  } else if (goma_ctx->fem.mp->DensityModel == DENSITY_CURE_SHRINKAGE) {
    // parameters
    dbl rho_l = goma_ctx->fem.mp->u_density[0];
    dbl rho_s = goma_ctx->fem.mp->u_density[1];
    dbl alpha_m = goma_ctx->fem.mp->u_density[2];
    dbl alpha_g = goma_ctx->fem.mp->u_density[3];
    dbl cure_enable = goma_ctx->fem.mp->u_density[4];

    int ConstitutiveEquation =
        goma_ctx->fem.gn_glob[goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->mn]->ConstitutiveEquation;
    int species_no = 0;

    if (ConstitutiveEquation == CURE || ConstitutiveEquation == EPOXY ||
        ConstitutiveEquation == FILLED_EPOXY || ConstitutiveEquation == FOAM_PMDI_10) {
      species_no =
          goma_ctx->fem.gn_glob[goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->mn]->cure_species_no;
    } else {
      GOMA_EH(GOMA_ERROR, "Unknown constitutive equation for density model CURE_SHRINKAGE");
    }

    if (goma_ctx->fem.fv->c[species_no] >= cure_enable) {
      rho = rho_l + ((rho_s - rho_l) / (alpha_m - alpha_g)) * (goma_ctx->fem.fv->c[0] - alpha_g);

      if (d_rho != NULL) {
        var = MASS_FRACTION;
        if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][var]) {
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[var]; j++) {
            d_rho->C[species_no][j] =
                ((rho_s - rho_l) / (alpha_m - alpha_g)) * goma_ctx->fem.bf[var]->phi[j];
          }
        }
      }
    } else {
      rho = rho_l + ((rho_s - rho_l) / (alpha_m - alpha_g)) * (cure_enable - alpha_g);
    }
  } else {
    GOMA_EH(GOMA_ERROR, "Unrecognized density model");
  }

  if (goma_ctx->fem.ls != NULL && goma_ctx->fem.mp->mp2nd != NULL &&
      (goma_ctx->fem.mp->DensityModel != LEVEL_SET) &&
      (goma_ctx->fem.mp->mp2nd->DensityModel == CONSTANT)) {
    double factor;

    if (d_rho == NULL) {
      /* kludge for solidification tracking with phase function 0 */
      for (imtrx = 0; imtrx < goma_ctx->fem.upd->Total_Num_Matrices; imtrx++) {
        if (goma_ctx->fem.pfd != NULL && goma_ctx->fem.pd->e[imtrx][R_EXT_VELOCITY]) {
          ls_old = goma_ctx->fem.ls;
          goma_ctx->fem.ls = goma_ctx->fem.pfd->ls[0];
          rho = ls_modulate_property(rho, goma_ctx->fem.mp->mp2nd->density_phase[0],
                                     goma_ctx->fem.ls->Length_Scale,
                                     (double)goma_ctx->fem.mp->mp2nd->densitymask[0],
                                     (double)goma_ctx->fem.mp->mp2nd->densitymask[1], NULL, &factor,
                                     goma_ctx->fem.mp->mp2nd->density_lsi_interp_method);
          goma_ctx->fem.ls = ls_old;
        }
      }
      rho = ls_modulate_property(rho, goma_ctx->fem.mp->mp2nd->density,
                                 goma_ctx->fem.ls->Length_Scale,
                                 (double)goma_ctx->fem.mp->mp2nd->densitymask[0],
                                 (double)goma_ctx->fem.mp->mp2nd->densitymask[1], NULL, &factor,
                                 goma_ctx->fem.mp->mp2nd->density_lsi_interp_method);
    } else {
      /* kludge for solidification tracking with phase function 0 */
      if (goma_ctx->fem.pfd != NULL &&
          goma_ctx->fem.pd->e[goma_ctx->fem.pg->imtrx][R_EXT_VELOCITY]) {
        ls_old = goma_ctx->fem.ls;
        goma_ctx->fem.ls = goma_ctx->fem.pfd->ls[0];
        rho = ls_modulate_property(rho, goma_ctx->fem.mp->mp2nd->density_phase[0],
                                   goma_ctx->fem.ls->Length_Scale,
                                   (double)goma_ctx->fem.mp->mp2nd->densitymask[0],
                                   (double)goma_ctx->fem.mp->mp2nd->densitymask[1], d_rho->F,
                                   &factor, goma_ctx->fem.mp->mp2nd->density_lsi_interp_method);
        goma_ctx->fem.ls = ls_old;
      }
      rho = ls_modulate_property(rho, goma_ctx->fem.mp->mp2nd->density,
                                 goma_ctx->fem.ls->Length_Scale,
                                 (double)goma_ctx->fem.mp->mp2nd->densitymask[0],
                                 (double)goma_ctx->fem.mp->mp2nd->densitymask[1], d_rho->F, &factor,
                                 goma_ctx->fem.mp->mp2nd->density_lsi_interp_method);
      if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][MASS_FRACTION]) {
        for (w = 0; w < goma_ctx->fem.pd->Num_Species_Eqn; w++) {
          for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[MASS_FRACTION]; j++) {
            d_rho->C[w][j] *= factor;
          }
        }
      }

      if (goma_ctx->fem.pd->v[goma_ctx->fem.pg->imtrx][TEMPERATURE]) {
        for (j = 0; j < goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->dof[TEMPERATURE]; j++) {
          d_rho->T[j] *= factor;
        }
      }
    }
  }

  return (rho);
}
