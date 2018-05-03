#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


#include "std.h"
#include "rf_fem_const.h"
#include "rf_fem.h"
#include "rf_masks.h"
#include "rf_io_const.h"
#include "rf_io_structs.h"
#include "rf_io.h"
#include "rf_mp.h"
#include "el_elm.h"
#include "el_geom.h"
#include "rf_bc_const.h"
#include "rf_solver_const.h"
#include "rf_fill_const.h"
#include "rf_vars_const.h"
#include "mm_mp_const.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "mm_as.h"

#include "mm_eh.h"

#include "mm_fill_ls.h"

#include "mm_mp_structs.h"
#include "mm_mp_const.h"
#include "mm_mp.h"

#include "mm_fill_terms.h"
/* GOMA include files */
#define _MM_FILL_POPULATION_C

#include "mm_fill_population.h"
#include "sl_aux.h"
#include "goma.h"



extern FSUB_TYPE dsyev_(char *JOBZ, char *UPLO, int *N, double *A, int *LDA,
			double *W, double *WORK, int *LWORK, int *INFO,
			int len_jobz, int len_uplo);

static void
compute_nodes_weights(int N, double Jac[N+1][N+1], double *weights, double *nodes, double *moments)
{
  int LDA = N;
  int i,j;

  int INFO;
  int LWORK = 20;
  double WORK[LWORK];
  memset(WORK, 0, sizeof(double)*LWORK);

  double A[N*N];
  memset(A, 0.0, sizeof(double)*N*N);

  // convert to column major
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      A[i*N + j] = Jac[j][i];
    }
  }

  double W[N];

  // eig solver
  dsyev_("V", "U", &N, A, &LDA, W, WORK, &LWORK, &INFO, 1, 1);

  double U[N][N];

  // transpose (revert to row major)
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
    	U[i][j] = A[j*N + i];
    }
  }


  // Eigenvalues of conformation tensor
  for (i = 0; i < N; i++) {
    weights[i] = U[0][i]*U[0][i]*moments[0];
    nodes[i] = W[i];
  }

}

void wheeler_algorithm(int N, double *moments, double *weights, double *nodes)
{
  double P[2*N+1][2*N+1];
  double Jac[N+1][N+1];
  double a[N+1];
  double b[N+1];

  for (int i = 0; i < 2*N+1; i++) {
    for (int j = 0; j < 2*N+1; j++) {
      P[i][j] = 0;
    }
  }

  for (int i = 0; i < N+1; i++) {
    a[i] = 0;
    b[i] = 0;
    for (int j = 0; j < N+1; j++) {
      Jac[i][j] = 0;
    }
  }

  for (int i = 0; i <= 2*N-1; i++) {
    P[1][i] = moments[i];
  }

  a[0] = moments[1]/moments[0];
  b[0] = 0;

  for (int i = 0; i < N-1; i++) {
    for (int j = i; j < (2*N - i -2); j++) {
      P[i+2][j+1] = P[i+1][j+2] - a[i]*P[i+1][j+1] - b[i]*P[i][j+1];
    }
    a[i+1] = -P[i+1][i+1]/P[i+1][i] + P[i+2][i+2]/P[i+2][i+1];
    b[i+1] = P[i+2][i+1] / P[i+1][i];
  }

  for (int i = 0; i < N-1; i++) {
    Jac[i][i] = a[i];
    Jac[i][i+1] = -sqrt(fabs(b[i+1]));
    Jac[i+1][i] = -sqrt(fabs(b[i+1]));
  }

  for (int i = 0; i < N-1; i++) {
    weights[i] = 0;
  }

  compute_nodes_weights(N, Jac, weights, nodes, moments);
}

int get_foam_pbe_indices(int *index_W, int *index_OH, int *index_BA_l,
			  int *index_BA_g, int *index_CO2_l, int *index_CO2_g)
{
  int w;
  int species_W = -1;
  int species_OH = -1;
  int species_BA_l = -1;
  int species_BA_g = -1;
  int species_CO2_l = -1;
  int species_CO2_g = -1;

  /* find equation that has extent of reaction */
  for (w = 0; w < pd->Num_Species_Eqn; w++) {
    switch (mp->SpeciesSourceModel[w]) {
    case FOAM_PBE_WATER:
      if (species_W == -1) {
	species_W = w;
      } else {
	EH(-1, "Error expected only one FOAM_PBE_WATER species");
      }
      break;
    case FOAM_PBE_OH:
      if (species_OH == -1) {
	species_OH = w;
      } else {
	EH(-1, "Error expected only one FOAM_PBE_OH species");
      }
      break;
    case FOAM_PBE_BA_L:
      if (species_BA_l == -1) {
	species_BA_l = w;
      } else {
	EH(-1, "Error expected only one FOAM_PBE_BA_L species");
      }
      break;
    case FOAM_PBE_BA_G:
      if (species_BA_g == -1) {
	species_BA_g = w;
      } else {
	EH(-1, "Error expected only one FOAM_PBE_BA_G species");
      }
      break;
    case FOAM_PBE_CO2_L:
      if (species_CO2_l == -1) {
	species_CO2_l = w;
      } else {
	EH(-1, "Error expected only one FOAM_PBE_CO2_L species");
      }
      break;
    case FOAM_PBE_CO2_G:
      if (species_CO2_g == -1) {
	species_CO2_g = w;
      } else {
	EH(-1, "Error expected only one FOAM_PBE_CO2_G species");
      }
      break;
    default:
      break;
    }
  }

  if (species_W == -1) {
    EH(-1, "Error expected FOAM_PBE_WATER species");
    return -1;
  }

  if (species_OH == -1) {
    EH(-1, "Error expected FOAM_PBE_OH species");
    return -1;
  }

  if (species_BA_l == -1) {
    EH(-1, "Error expected FOAM_PBE_BA_L species");
    return -1;
  }

  if (species_BA_g == -1) {
    EH(-1, "Error expected FOAM_PBE_BA_G species");
    return -1;
  }

  if (species_CO2_l == -1) {
    EH(-1, "Error expected FOAM_PBE_CO2_L species");
    return -1;
  }

  if (species_CO2_g == -1) {
    EH(-1, "Error expected FOAM_PBE_CO2_G species");
    return -1;
  }

  if (index_W != NULL) {
    *index_W = species_W;
  }

  if (index_OH != NULL) {
    *index_OH = species_OH;
  }

  if (index_BA_l != NULL) {
    *index_BA_l = species_BA_l;
  }

  if (index_BA_g != NULL) {
    *index_BA_g = species_BA_g;
  }

  if (index_CO2_l != NULL) {
    *index_CO2_l = species_CO2_l;
  }

  if (index_CO2_g != NULL) {
    *index_CO2_g = species_CO2_g;
  }

  return 0;
}

double
foam_pbe_heat_source(HEAT_SOURCE_DEPENDENCE_STRUCT *d_h,
		     double tt,	/* parameter to vary time integration from
				 * explicit (tt = 1) to implicit (tt = 0) */
		     double dt)	/* current time step size */
{
  int species_W = -1;
  int species_OH = -1;
  int species_BA_l = -1;

  double h = 0.;
  int var;
  int j;

  /* Begin Execution */

  int err;

  err = get_foam_pbe_indices(&species_W, &species_OH, &species_BA_l,
			     NULL, NULL, NULL);
  if (err) return 0;

  double C0_OH = mp->u_species_source[species_OH][0];
  double Delta_H_OH = mp->u_species_source[species_OH][3];

  double C0_W = mp->u_species_source[species_W][0];
  double Delta_H_W = mp->u_species_source[species_W][3];

  double Lambda = mp->u_species_source[species_BA_l][1];

  h = -Delta_H_W * C0_W * fv_dot->c[species_W] - Delta_H_OH * C0_OH * fv_dot->c[species_OH] + Lambda * fv_dot->c[species_BA_l];

  var = MASS_FRACTION;
  if ( d_h != NULL && pd->v[pg->imtrx][var] )
    {
      for (j=0; j<ei[pg->imtrx]->dof[var]; j++)
	{
	  d_h->C[species_W][j] += -Delta_H_W * C0_W * (1 + 2. * tt)/dt *bf[var]->phi[j];
	  d_h->C[species_OH][j] += -Delta_H_OH * C0_OH * (1 + 2. * tt)/dt *bf[var]->phi[j];
	  d_h->C[species_BA_l][j] += Lambda * (1 + 2. * tt)/dt *bf[var]->phi[j];
	}
    }

  return (h);
}

int foam_pbe_growth_rate(double growth_rate[MAX_CONC], double d_growth_rate_dc[MAX_CONC][MDE], double d_growth_rate_dT[MAX_CONC][MDE])
{
  double G0, T0;
  int species_BA_l;
  int species_CO2_l;
  int err;
  int w,j;
  int var = MASS_FRACTION;
  double M_BA;
  double M_NCO;

  err = get_foam_pbe_indices(NULL, NULL, &species_BA_l,
			     NULL, &species_CO2_l, NULL);

  if (err) return -1;

  for (w = 0; w < MAX_CONC; w++) {
    growth_rate[w] = 0;
    for (j = 0; j < MDE; j++) {
      d_growth_rate_dc[w][j] = 0.0;
      d_growth_rate_dT[w][j] = 0.0;
    }
  }

  M_BA = mp->u_species_source[species_BA_l][0];
  G0 = mp->u_species_source[species_BA_l][2];
  T0 = mp->u_species_source[species_BA_l][3];
  M_NCO = mp->u_species_source[species_BA_l][4];
  for (w = 0; w < pd->Num_Species_Eqn; w++) {

    if (w == species_BA_l) {
      if (mp->PBE_BA_Type == PBE_R_11) {
	double xBL = -0.012 * (fv->T - 300) + 0.5;
	double d_xBL_dT = -0.012;
	if (xBL < 0) {
	  xBL = 0;
	  d_xBL_dT = 0;
	}
	double R11_max = 0;
	double d_R11_max_dT = 0;
	if (xBL != 0) {
	  R11_max = (xBL/(1-xBL)) * (M_BA / M_NCO);
	  d_R11_max_dT = (M_BA / M_NCO)*(d_xBL_dT*(1-xBL) - xBL*(-d_xBL_dT))/((1-xBL)*(1-xBL));
	}



	if (fv->c[species_BA_l] > R11_max && R11_max > PBE_FP_SMALL) {
	  growth_rate[species_BA_l] = G0 * (fv->c[species_BA_l] - R11_max)/ R11_max;
	  for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
	    d_growth_rate_dc[species_BA_l][j] = G0 * bf[var]->phi[j] / R11_max;
	  }

	  for (j = 0; j < ei[pg->imtrx]->dof[TEMPERATURE]; j++) {
	    d_growth_rate_dT[species_BA_l][j] = -G0 * bf[TEMPERATURE]->phi[j] * d_R11_max_dT / (R11_max*R11_max);
	  }
	}
      } else if (mp->PBE_BA_Type == PBE_N_PENTANE) {
	const double a = 0.0064;
	const double h = 0.0551;
	const double q = 17.8;
	double n_pentane_max = a + h * exp(-(fv->T - T0)*(fv->T - T0)/(2*q*q));
	double d_npentane_max_dT = h * (-fv->T + T0) *  exp(-(fv->T - T0)*(fv->T - T0)/(2*q*q)) / (q*q);
	if (fv->c[species_BA_l] > n_pentane_max) {
	  growth_rate[species_BA_l] = G0 * (fv->c[species_BA_l] - n_pentane_max)/n_pentane_max;
	  for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
	    d_growth_rate_dc[species_BA_l][j] = G0 * bf[var]->phi[j] / n_pentane_max;
	  }

	  for (j = 0; j < ei[pg->imtrx]->dof[TEMPERATURE]; j++) {
	    d_growth_rate_dT[species_BA_l][j] = G0 * (-fv->c[species_BA_l] * d_npentane_max_dT * bf[TEMPERATURE]->phi[j])
	      / (n_pentane_max*n_pentane_max);
	  }
	}
      } else {
	EH(-1, "Unknown PBE BA Type");
	return -1;
      }

    } else if (w == species_CO2_l) {
      double CO2_max = 4e-4;
      if (fv->c[species_CO2_l] > CO2_max) {
	growth_rate[species_CO2_l] = G0 * (fv->c[species_CO2_l] - CO2_max) / CO2_max;
	for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
	  d_growth_rate_dc[species_CO2_l][j] = G0 * bf[var]->phi[j] / CO2_max;
	}

	for (j = 0; j < ei[pg->imtrx]->dof[TEMPERATURE]; j++) {
	  d_growth_rate_dT[species_CO2_l][j] = 0;
	}
      }
    }

  }

  return 0;
}

int foam_pmdi_growth_rate(double growth_rate[MAX_CONC], double d_growth_rate_dc[MAX_CONC][MDE], double d_growth_rate_dT[MAX_CONC][MDE])
{
  int wCO2Liq;
  int wCO2Gas;
  int wH2O;
  int w;

  wCO2Liq = -1;
  wCO2Gas = -1;
  wH2O = -1;
  for (w = 0; w < pd->Num_Species; w++) {
    switch (mp->SpeciesSourceModel[w]) {
    case FOAM_PMDI_10_CO2_LIQ:
      wCO2Liq = w;
      break;
    case FOAM_PMDI_10_CO2_GAS:
      wCO2Gas = w;
      break;
    case FOAM_PMDI_10_H2O:
      wH2O = w;
      break;
    default:
      break;
    }
  }

  if (wCO2Liq == -1) {
    EH(-1, "Expected a Species Source of FOAM_PMDI_10_CO2_LIQ");
    return -1;
  } else if (wH2O == -1) {
    EH(-1, "Expected a Species Source of FOAM_PMDI_10_H2O");
    return -1;
  } else if (wCO2Gas == -1) {
    EH(-1, "Expected a Species Source of FOAM_PMDI_10_CO2_GAS");
    return -1;
  }

  double G0;

  G0 = mp->u_moment_source[0];

  int j;

  for (w = 0; w < MAX_CONC; w++) {
    growth_rate[w] = 0;
    for (j = 0; j < MDE; j++) {
      d_growth_rate_dc[w][j] = 0.0;
      d_growth_rate_dT[w][j] = 0.0;
    }
  }

  if (mp->DensityModel == DENSITY_FOAM_PMDI_10)
    {
      int var, j;
      int w;

      double M_CO2 = mp->u_density[0];
      double rho_liq = mp->u_density[1];

      var = MASS_FRACTION;

      for (w = 0; w < pd->Num_Species_Eqn; w++) {

	if (w == wCO2Liq) {
	  double CO2_max = 4.4e-4;
	  double mf = fv->c[wCO2Liq] * M_CO2 / rho_liq;
	  double dmfdC = M_CO2 / rho_liq;

	  if (mf > CO2_max) {
	    growth_rate[wCO2Liq] = G0 * (mf - CO2_max) / CO2_max;
	    for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
	      d_growth_rate_dc[wCO2Liq][j] = G0 * dmfdC * bf[var]->phi[j] / CO2_max;
	    }

	    for (j = 0; j < ei[pg->imtrx]->dof[TEMPERATURE]; j++) {
	      d_growth_rate_dT[wCO2Liq][j] = 0;
	    }
	  }
	}
      }
    }
  else
    {
      EH(-1, "Expected DENSITY_FOAM_PMDI_10 for growth rate");
      return -1;
    }

  return 0;
}


double
foam_pbe_conductivity( CONDUCTIVITY_DEPENDENCE_STRUCT *d_k,
		       dbl time )
{
  DENSITY_DEPENDENCE_STRUCT d_rho_struct;
  DENSITY_DEPENDENCE_STRUCT *d_rho = &d_rho_struct;

  double rho = density(d_rho, time);

  double lambda = 0;


  if (rho > 48) {
    lambda = (8.7006e-8)*rho*rho + (8.4674e-5)*rho + 1.16e-2;
  } else {
    lambda = (9.3738e-6)*rho*rho - (7.3511e-4)*rho + 2.965e-2;
  }

  if (d_k != NULL) {

    int mom;
    int j;
    for (mom = 0; mom < MAX_MOMENTS; mom++) {
      for (j = 0; j < MDE; j++) {
	d_k->moment[mom][j] = 0.0;
      }
    }

    if (pd->v[pg->imtrx][TEMPERATURE]) {
      for (j = 0; j < ei[pg->imtrx]->dof[TEMPERATURE]; j++) {
	if (rho > 48) {
	  d_k->T[j] = 2*(8.7006e-8)*rho*d_rho->T[j] + (8.4674e-5)*d_rho->T[j];
	} else {
	  d_k->T[j] = 2*(9.3738e-6)*rho*d_rho->T[j] - (7.3511e-4)*d_rho->T[j];
	}
      }
    }


    if (pd->v[pg->imtrx][MOMENT1]) {
      for (j = 0; j < ei[pg->imtrx]->dof[MOMENT1]; j++) {
	if (rho > 48) {
	  d_k->moment[1][j] = 2*(8.7006e-8)*rho*d_rho->moment[1][j] + (8.4674e-5)*d_rho->moment[1][j];
	} else {
	  d_k->moment[1][j] = 2*(9.3738e-6)*rho*d_rho->moment[1][j] - (7.3511e-4)*d_rho->moment[1][j];
	}
      }
    }
  }

  return lambda;
}

void foam_pbe_conversion_water(struct Species_Conservation_Terms *st,
			       double time,
			       double tt,
			       double dt)
{
  int species_W = -1;
  int species_OH = -1;

  double source = 0.;
  int var;
  int j;

  int err;
  err = get_foam_pbe_indices(&species_W, &species_OH, NULL,
			     NULL, NULL, NULL);
  if (err) return;


  double A_W = mp->u_species_source[species_W][1];
  double E_W = mp->u_species_source[species_W][2];
  double Rgas_const = mp->u_density[2];
  double gelling_point = mp->u_species_source[species_OH][5];

  double coeff = A_W * exp(-E_W/(Rgas_const*fv->T));

  source = coeff * (1 - fv->c[species_W]);
  st->MassSource[species_W] = source;

  if (fv->c[species_OH] > gelling_point) {
    source = 0;
    st->MassSource[species_W] = source;
    return;
  }

  if (af->Assemble_Jacobian) {
    var = MASS_FRACTION;
    for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
      st->d_MassSource_dc[species_W][species_W][j] = - coeff * bf[var]->phi[j];
    }

    var = TEMPERATURE;
    if (pd->v[pg->imtrx][var]) {
      for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
	st->d_MassSource_dT[species_W][j] = (E_W/(Rgas_const*fv->T*fv->T)) * coeff * bf[var]->phi[j] * (1 - fv->c[species_W]);
      }
    }
  }
}

void foam_pbe_conversion_OH(struct Species_Conservation_Terms *st,
			    double time,
			    double tt,
			    double dt)
{
  int species_W = -1;
  int species_OH = -1;

  double source = 0.;
  int var;
  int j;

  int err;
  err = get_foam_pbe_indices(&species_W, &species_OH, NULL,
			     NULL, NULL, NULL);
  if (err) return;


  double A_OH = mp->u_species_source[species_OH][1];
  double E_OH = mp->u_species_source[species_OH][2];
  double Rgas_const = mp->u_density[2];
  double C0_NCO = mp->u_species_source[species_OH][4];
  double C0_W = mp->u_species_source[species_W][0];
  double C0_OH = mp->u_species_source[species_OH][0];
  double gelling_point = mp->u_species_source[species_OH][5];
  double coeff = A_OH * exp(-E_OH/(Rgas_const*fv->T));


  double X_OH = fv->c[species_OH];
  double X_W = fv->c[species_W];

  source = coeff * C0_OH * (C0_NCO/C0_OH
			    - 2*(C0_W/C0_OH)*X_W
			    - X_OH*(1+C0_NCO/C0_OH)
			    + 2*(C0_W/C0_OH)*X_W*X_OH
			    + X_OH * X_OH);

  st->MassSource[species_OH] = source;

  if (fv->c[species_OH] > gelling_point) {
    source = 0;
    st->MassSource[species_OH] = source;
    return;
  }

  if (af->Assemble_Jacobian) {
    var = MASS_FRACTION;
    for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
      st->d_MassSource_dc[species_OH][species_W][j] = coeff * C0_OH * (-2*(C0_W/C0_OH)*bf[var]->phi[j]
								       + 2*(C0_W/C0_OH)*bf[var]->phi[j]*X_OH);

      st->d_MassSource_dc[species_OH][species_OH][j] = coeff * C0_OH * (-(1+C0_NCO/C0_OH)*bf[var]->phi[j]
									+ 2*(C0_W/C0_OH)*X_W*bf[var]->phi[j]
									+ 2*X_OH *bf[var]->phi[j]);

    }

    var = TEMPERATURE;
    if (pd->v[pg->imtrx][var]) {
      for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
	st->d_MassSource_dT[species_OH][j] = (E_OH/(Rgas_const*fv->T*fv->T)) * coeff * bf[var]->phi[j]
	  * C0_OH * (C0_NCO/C0_OH
		     - 2*(C0_W/C0_OH)*X_W
		     - X_OH*(1+C0_NCO/C0_OH)
		     + 2*(C0_W/C0_OH)*X_W*X_OH
		     + X_OH * X_OH);

      }
    }
  }
}

void foam_pbe_ba_gas_source(struct Species_Conservation_Terms *st,
			    double time,
			    double tt,
			    double dt)
{
  int species_BA_l;
  int species_BA_g;
  int species_CO2_l;
  int err = 0;
  int var;
  int j;

  struct moment_growth_rate *MGR = NULL;

  err = get_foam_pbe_indices(NULL, NULL, &species_BA_l,
			     &species_BA_g, &species_CO2_l, NULL);
  if (err) return;

  MGR = calloc(sizeof(struct moment_growth_rate), 1);
  err = get_moment_growth_rate_term(MGR);

  double source = 0;
  double rho_foam = mp->u_density[0];
  double ref_press = mp->u_density[1];
  double Rgas_const = mp->u_density[2];
  double M_BA = mp->u_species_source[species_BA_l][0];

  source = MGR->G[species_BA_l][1] * ref_press/(Rgas_const*fv->T)*M_BA/rho_foam;
  st->MassSource[species_BA_g] = source;

  if (af->Assemble_Jacobian) {
    var = MASS_FRACTION;
    for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
      st->d_MassSource_dc[species_BA_g][species_BA_l][j] = MGR->d_G_dC[species_BA_l][1][j] * ref_press/(Rgas_const*fv->T)*M_BA/rho_foam;
    }

    var = TEMPERATURE;
    if (pd->v[pg->imtrx][var]) {
      for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
	st->d_MassSource_dT[species_BA_g][j] = -MGR->G[species_BA_l][1] *
	  (ref_press / (Rgas_const * fv->T * fv->T) * M_BA/rho_foam) * bf[var]->phi[j];
      }
    }
  }

  free(MGR);
}

void foam_pbe_ba_liquid_source(struct Species_Conservation_Terms *st,
			    double time,
			    double tt,
			    double dt)
{
  int species_BA_l;
  int species_CO2_l;
  int err = 0;
  int var;
  int j;

  struct moment_growth_rate *MGR = NULL;

  err = get_foam_pbe_indices(NULL, NULL, &species_BA_l,
			     NULL, &species_CO2_l, NULL);
  if (err) return;

  MGR = calloc(sizeof(struct moment_growth_rate), 1);
  err = get_moment_growth_rate_term(MGR);

  double source = 0;
  double rho_foam = mp->u_density[0];
  double ref_press = mp->u_density[1];
  double Rgas_const = mp->u_density[2];
  double M_BA = mp->u_species_source[species_BA_l][0];

  source = -MGR->G[species_BA_l][1] * ref_press/(Rgas_const*fv->T)*M_BA/rho_foam;

  st->MassSource[species_BA_l] = source;

  if (af->Assemble_Jacobian) {
    var = MASS_FRACTION;
    for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
      st->d_MassSource_dc[species_BA_l][species_BA_l][j] = -MGR->d_G_dC[species_BA_l][1][j] * ref_press/(Rgas_const*fv->T)*M_BA/rho_foam;
    }

    var = TEMPERATURE;
    if (pd->v[pg->imtrx][var]) {
      for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
	st->d_MassSource_dT[species_BA_l][j] = MGR->G[species_BA_l][1] *
	  (ref_press / (Rgas_const * fv->T * fv->T) * M_BA/rho_foam) * bf[var]->phi[j];
      }
    }
  }

  free(MGR);
}

void foam_pbe_co2_gas_source(struct Species_Conservation_Terms *st,
			     double time,
			     double tt,
			     double dt)
{
  int species_W;
  int species_BA_l;
  int species_CO2_l;
  int species_CO2_g;
  int err = 0;
  int var;
  int j;

  struct moment_growth_rate *MGR = NULL;

  err = get_foam_pbe_indices(&species_W, NULL, &species_BA_l,
			     NULL, &species_CO2_l, &species_CO2_g);
  if (err) return;

  MGR = calloc(sizeof(struct moment_growth_rate), 1);
  err = get_moment_growth_rate_term(MGR);

  double source = 0;
  double rho_foam = mp->u_density[0];
  double ref_press = mp->u_density[1];
  double Rgas_const = mp->u_density[2];
  double M_CO2 = mp->u_species_source[species_CO2_l][0];

  source = MGR->G[species_CO2_l][1] * ref_press/(Rgas_const*fv->T)*M_CO2/rho_foam;

  st->MassSource[species_CO2_g] = source;

  if (af->Assemble_Jacobian) {
    var = MASS_FRACTION;
    for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
      st->d_MassSource_dc[species_CO2_g][species_CO2_l][j] = MGR->d_G_dC[species_CO2_l][1][j] * ref_press/(Rgas_const*fv->T)*M_CO2/rho_foam;
    }

    var = TEMPERATURE;
    if (pd->v[pg->imtrx][var]) {
      for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
	st->d_MassSource_dT[species_CO2_g][j] = -MGR->G[species_CO2_l][1] *
	  (ref_press / (Rgas_const * fv->T * fv->T) * M_CO2/rho_foam) * bf[var]->phi[j];
      }
    }
  }

  free(MGR);
}

void foam_pbe_co2_liquid_source(struct Species_Conservation_Terms *st,
			    double time,
			    double tt,
			    double dt)
{
  int species_W;
  int species_BA_l;
  int species_CO2_l;
  int err = 0;
  int var;
  int j;

  struct moment_growth_rate *MGR = NULL;

  err = get_foam_pbe_indices(&species_W, NULL, &species_BA_l,
			     NULL, &species_CO2_l, NULL);
  if (err) return;

  MGR = calloc(sizeof(struct moment_growth_rate), 1);
  err = get_moment_growth_rate_term(MGR);

  double source = 0;
  double rho_foam = mp->u_density[0];
  double ref_press = mp->u_density[1];
  double Rgas_const = mp->u_density[2];
  double C0_W = mp->u_species_source[species_W][0];
  double M_CO2 = mp->u_species_source[species_CO2_l][0];

  source = (C0_W*fv_dot->c[species_W] - MGR->G[species_CO2_l][1] * ref_press/(Rgas_const*fv->T))*M_CO2/rho_foam;

  st->MassSource[species_CO2_l] = source;

  if (af->Assemble_Jacobian) {
    var = MASS_FRACTION;
    for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
      st->d_MassSource_dc[species_CO2_l][species_CO2_l][j] = -MGR->d_G_dC[species_CO2_l][1][j] * ref_press/(Rgas_const*fv->T)*M_CO2/rho_foam;

      st->d_MassSource_dc[species_CO2_l][species_W][j] = (C0_W*(1.+2.*tt)/dt*bf[var]->phi[j])*M_CO2/rho_foam;
    }

    var = TEMPERATURE;
    if (pd->v[pg->imtrx][var]) {
      for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
	st->d_MassSource_dT[species_CO2_l][j] = MGR->G[species_CO2_l][1] *
	  (ref_press / (Rgas_const * fv->T * fv->T) * M_CO2/rho_foam) * bf[var]->phi[j];
      }
    }
  }


  free(MGR);
}

int get_moment_growth_rate_term(struct moment_growth_rate *MGR)
{
  int nnodes = 2; // currently hardcoded for 2 Nodes (4 Moments)
  double weights[nnodes], nodes[nnodes];
  double growth_rate[MAX_CONC];
  double d_growth_rate_dc[MAX_CONC][MDE];
  double d_growth_rate_dT[MAX_CONC][MDE];

  if (!pd->gv[MOMENT0] ||
      !pd->gv[MOMENT1] ||
      !pd->gv[MOMENT2] ||
      !pd->gv[MOMENT3]) {
    EH(-1, "Expected Moment equations for moment growth rate");
    return -1;
  }

  /* Get quad weights and nodes */
  wheeler_algorithm(nnodes, fv_old->moment, weights, nodes);

  switch (mp->MomentSourceModel) {
  case FOAM_PBE:
    {
      int species_BA_l;
      int species_CO2_l;
      int species_OH;
      int err;
      int w,j;

      for (w = 0; w < MAX_CONC; w++) {
	for (j = 0; j < MDE; j++) {
	  growth_rate[w] = 0.0;
	  d_growth_rate_dc[w][j] = 0.0;
	  d_growth_rate_dT[w][j] = 0.0;
	}
      }


      err = get_foam_pbe_indices(NULL, &species_OH, &species_BA_l,
				 NULL, &species_CO2_l, NULL);
      if (err) return err;

      double gelling_point = mp->u_species_source[species_OH][5];
      if (fv->c[species_OH] > gelling_point) {
	return 0;
      }


      foam_pbe_growth_rate(growth_rate, d_growth_rate_dc, d_growth_rate_dT);

      // currently only compute G_wi terms for CO2_l and BA_l

      MGR->G[species_BA_l][0] = 0;
      MGR->G[species_CO2_l][0] = 0;
      for (int k = 1; k < 2*nnodes; k++) {
	for (int alpha = 0; alpha < nnodes; alpha++) {
	  double coeff = k * weights[alpha] * pow(nodes[alpha], k-1);
	  MGR->G[species_BA_l][k] = coeff*growth_rate[species_BA_l];
	  MGR->G[species_CO2_l][k] = coeff * growth_rate[species_CO2_l];
	  if (af->Assemble_Jacobian) {
	    if (pd->v[pg->imtrx][MASS_FRACTION]) {
	      for (j = 0; j < ei[pg->imtrx]->dof[MASS_FRACTION]; j++) {
		MGR->d_G_dC[species_BA_l][k][j] = coeff * d_growth_rate_dc[species_BA_l][j];
		MGR->d_G_dC[species_CO2_l][k][j] = coeff * d_growth_rate_dc[species_CO2_l][j];
	      }
	    }

	    if (pd->v[pg->imtrx][TEMPERATURE]) {
	      for (j = 0; j < ei[pg->imtrx]->dof[TEMPERATURE]; j++) {
		MGR->d_G_dT[species_BA_l][k][j] = coeff * d_growth_rate_dT[species_BA_l][j];
		MGR->d_G_dT[species_CO2_l][k][j] = coeff * d_growth_rate_dT[species_CO2_l][j];
	      }
	    }
	  }
	}
      }
      return 0;
    }
    break;
  case FOAM_PMDI_10:
    {
      int wCO2Liq;
      int wCO2Gas;
      int wH2O;
      int w;
      int j;

      wCO2Liq = -1;
      wCO2Gas = -1;
      wH2O = -1;
      for (w = 0; w < pd->Num_Species; w++) {
	switch (mp->SpeciesSourceModel[w]) {
	case FOAM_PMDI_10_CO2_LIQ:
	  wCO2Liq = w;
	  break;
	case FOAM_PMDI_10_CO2_GAS:
	  wCO2Gas = w;
	  break;
	case FOAM_PMDI_10_H2O:
	  wH2O = w;
	  break;
	default:
	  break;
	}
      }

      if (wCO2Liq == -1) {
	EH(-1, "Expected a Species Source of FOAM_PMDI_10_CO2_LIQ");
	return -1;
      } else if (wH2O == -1) {
	EH(-1, "Expected a Species Source of FOAM_PMDI_10_H2O");
	return -1;
      } else if (wCO2Gas == -1) {
	EH(-1, "Expected a Species Source of FOAM_PMDI_10_CO2_GAS");
	return -1;
      }

      // Coalescence rate
      double Beta0 = mp->u_moment_source[1];

      foam_pmdi_growth_rate(growth_rate, d_growth_rate_dc, d_growth_rate_dT);

      // currently only compute G_wi terms for CO2_l and BA_l
      MGR->G[wCO2Liq][0] = 0;
      for (int k = 0; k < 2*nnodes; k++) {
	MGR->S[k] = 0;
	MGR->G[wCO2Liq][k] = 0;
	for (int alpha = 0; alpha < nnodes; alpha++) {
	  double coeff = k * weights[alpha] * pow(nodes[alpha], k-1);
	  MGR->G[wCO2Liq][k] += coeff * growth_rate[wCO2Liq];

	  for (int beta = 0; beta < nnodes; beta++) {
	    double coalescence_kernel = Beta0*(nodes[alpha] + nodes[beta]);
	    double wa = weights[alpha];
	    double wb = weights[beta];
	    double na = nodes[alpha];
	    double nb = nodes[beta];
	    MGR->S[k] += wa*wb*(pow(na + nb, k) - pow(na,k) - pow(nb,k))*coalescence_kernel;
	  }

	  if (af->Assemble_Jacobian) {
	    if (pd->v[pg->imtrx][MASS_FRACTION]) {
	      for (j = 0; j < ei[pg->imtrx]->dof[MASS_FRACTION]; j++) {
		MGR->d_G_dC[wCO2Liq][k][j] = coeff * d_growth_rate_dc[wCO2Liq][j];
	      }
	    }

	    if (pd->v[pg->imtrx][TEMPERATURE]) {
	      for (j = 0; j < ei[pg->imtrx]->dof[TEMPERATURE]; j++) {
		MGR->d_G_dT[wCO2Liq][k][j] = coeff * d_growth_rate_dT[wCO2Liq][j];
	      }
	    }
	  }
	}
	MGR->S[k] *= 0.5;
      }
      return 0;
    }
    break;
  default:
    EH(-1, "Unknown moment source model");
    return -1;
    break;
  }
  return -1;
}

int
moment_source(double *msource, MOMENT_SOURCE_DEPENDENCE_STRUCT *d_msource)
{
  struct moment_growth_rate *MGR = NULL;
  switch (mp->MomentSourceModel) {
  case FOAM_PBE:
    {

      int species_BA_l;
      int species_CO2_l;
      int err = 0;
      int j;
      double H = 1;

      if (ls != NULL) {
	load_lsi(ls->Length_Scale);
	H = 1-lsi->H;
      }

      struct moment_growth_rate *MGR = NULL;

      err = get_foam_pbe_indices(NULL, NULL, &species_BA_l,
				 NULL, &species_CO2_l, NULL);
      if (err) return err;

      MGR = calloc(sizeof(struct moment_growth_rate), 1);
      if (H > PBE_FP_SMALL) {
	err = get_moment_growth_rate_term(MGR);
      }

      if (err) {
	free(MGR);
	return err;
      }

      for (int mom = 0; mom < MAX_MOMENTS; mom++) {
	msource[mom] = H*(MGR->G[species_BA_l][mom] + MGR->G[species_CO2_l][mom]);
	if (pd->v[pg->imtrx][MASS_FRACTION]) {
	  for (j = 0; j < ei[pg->imtrx]->dof[MASS_FRACTION]; j++) {
	    d_msource->C[mom][species_BA_l][j] = H*MGR->d_G_dC[species_BA_l][mom][j];
	    d_msource->C[mom][species_CO2_l][j] = H*MGR->d_G_dC[species_CO2_l][mom][j];
	  }
	}

	if (pd->v[pg->imtrx][TEMPERATURE]) {
	  for (j = 0; j < ei[pg->imtrx]->dof[TEMPERATURE]; j++) {
	    d_msource->T[mom][j] = H*(MGR->d_G_dT[species_BA_l][mom][j] +  MGR->d_G_dT[species_CO2_l][mom][j]);
	  }
	}
      }
      free(MGR);
    }
    break;
  case FOAM_PMDI_10:
    {
      int wCO2Liq;
      int wCO2Gas;
      int wH2O;
      int w;
      int j;

      wCO2Liq = -1;
      wCO2Gas = -1;
      wH2O = -1;
      for (w = 0; w < pd->Num_Species; w++) {
	switch (mp->SpeciesSourceModel[w]) {
	case FOAM_PMDI_10_CO2_LIQ:
	  wCO2Liq = w;
	  break;
	case FOAM_PMDI_10_CO2_GAS:
	  wCO2Gas = w;
	  break;
	case FOAM_PMDI_10_H2O:
	  wH2O = w;
	  break;
	default:
	  break;
	}
      }

      if (wCO2Liq == -1) {
	EH(-1, "Expected a Species Source of FOAM_PMDI_10_CO2_LIQ");
	return -1;
      } else if (wH2O == -1) {
	EH(-1, "Expected a Species Source of FOAM_PMDI_10_H2O");
	return -1;
      } else if (wCO2Gas == -1) {
	EH(-1, "Expected a Species Source of FOAM_PMDI_10_CO2_GAS");
	return -1;
      }

      double H = 1;

      if (ls != NULL) {
	load_lsi(ls->Length_Scale);
	H = 1-lsi->H;
      }
      int err = 0;
      MGR = calloc(sizeof(struct moment_growth_rate), 1);
      if (H > PBE_FP_SMALL) {
	err = get_moment_growth_rate_term(MGR);
      }

      if (err) {
	free(MGR);
	return err;
      }

      for (int mom = 0; mom < MAX_MOMENTS; mom++) {
	msource[mom] = H*(MGR->G[wCO2Liq][mom] + MGR->S[mom]);
	if (pd->v[pg->imtrx][MASS_FRACTION]) {
	  for (j = 0; j < ei[pg->imtrx]->dof[MASS_FRACTION]; j++) {
	    d_msource->C[mom][wCO2Liq][j] = H*MGR->d_G_dC[wCO2Liq][mom][j];

	  }
	}
      }
      free(MGR);
    }
    break;
  default:
    EH(-1, "Unknown moment source model");
    break;
  }

  return 0;
}


int
assemble_density() /*  time step size      */
{
  int i,j,a,b;
  int peqn, pvar;
  int var;

  int dim;
  int eqn;
  int status=0;

  double det_J;
  double h3;			/* Volume element (scale factors). */
  double wt_func;
  double wt;

  double source;

  dim   = pd->Num_Dim;


  /*
   * Bail out fast if there's nothing to do...
   */

  if ( ! pd->e[pg->imtrx][eqn = R_DENSITY_EQN] )
    {
      return(status);
    }

  peqn = upd->ep[pg->imtrx][eqn];

  wt = fv->wt;

  det_J = bf[eqn]->detJ;

  h3 = fv->h3;

  double rho = 0;
  double d_rho_dT[MDE];
  double d_rho_dMOM[MAX_MOMENTS][MDE];
  double d_rho_dC[MAX_CONC][MDE];

  memset(d_rho_dT, 0, sizeof(double)*MDE);
  memset(d_rho_dC, 0, sizeof(double)*MAX_CONC*MDE);
  memset(d_rho_dMOM, 0, sizeof(double)*MAX_MOMENTS*MDE);

  if (mp->DensityModel == DENSITY_FOAM_PBE_EQN) {
    int species_BA_g;
    int species_CO2_g;
    int species_CO2_l;
    int species_BA_l;
    int err;
    err = get_foam_pbe_indices(NULL, NULL, &species_BA_l, &species_BA_g, &species_CO2_l, &species_CO2_g);
    if (err) return 0;

    double M_BA = mp->u_species_source[species_BA_l][0];
    double M_CO2 = mp->u_species_source[species_CO2_l][0];
    double rho_bubble = 0;
    double rho_foam = mp->u_density[0];
    double ref_press = mp->u_density[1];
    double Rgas_const = mp->u_density[2];

    if (fv->c[species_BA_g] > PBE_FP_SMALL || fv->c[species_CO2_g] > PBE_FP_SMALL) {
      rho_bubble = (ref_press/(Rgas_const*fv->T)) *
	(fv->c[species_CO2_g]*M_CO2 + fv->c[species_BA_g]*M_BA)/(fv->c[species_CO2_g] + fv->c[species_BA_g]);
    }

    double inv_mom_frac = 1/(1 + fv->moment[1]);
    rho = rho_bubble*(fv->moment[1]*inv_mom_frac) + rho_foam*inv_mom_frac;

    var = TEMPERATURE;
    if (pd->v[pg->imtrx][var] )
      {
	for ( j=0; j<ei[pg->imtrx]->dof[var]; j++)
	  {
	    d_rho_dT[j] = (-rho_bubble)/fv->T * (fv->moment[1]*inv_mom_frac) * bf[var]->phi[j];
	  }
      }

    var = MOMENT1;
    if (pd->v[pg->imtrx][var] )
      {
	for ( j=0; j<ei[pg->imtrx]->dof[var]; j++)
	  {
	    d_rho_dMOM[1][j] = (rho_bubble * inv_mom_frac * inv_mom_frac -
			     rho_foam * inv_mom_frac * inv_mom_frac) * bf[var]->phi[j];
	  }
      }

    var = MASS_FRACTION;
    if (pd->v[pg->imtrx][var])
      {
	for (j = 0; j < ei[pg->imtrx]->dof[var]; j++)
	  {
	    if (fv->c[species_BA_g] > PBE_FP_SMALL || fv->c[species_CO2_g] > PBE_FP_SMALL) {
	      d_rho_dC[species_BA_g][j] = (fv->moment[1]*inv_mom_frac) * bf[var]->phi[j] * (ref_press/(Rgas_const*fv->T)) *
		((M_BA - M_CO2)*fv->c[species_CO2_g])/
		((fv->c[species_CO2_g] + fv->c[species_BA_g]) * (fv->c[species_CO2_g] + fv->c[species_BA_g]));

	      d_rho_dC[species_CO2_g][j] = (fv->moment[1]*inv_mom_frac) * bf[var]->phi[j] * (ref_press/(Rgas_const*fv->T)) *
		((M_CO2-M_BA)*fv->c[species_BA_g])/
		((fv->c[species_CO2_g] + fv->c[species_BA_g]) * (fv->c[species_CO2_g] + fv->c[species_BA_g]));
	    }
	  }
      }
  } else {
    EH(-1, "Unknown density model for assemble_density");
  }




  /*
   * Residuals_________________________________________________________________
   */

  if ( af->Assemble_Residual )
    {
      if ( pd->e[pg->imtrx][eqn] )
	{
	  for( i=0; i<ei[pg->imtrx]->dof[eqn]; i++)
	    {

	      wt_func = bf[eqn]->phi[i];

	      source = 0.0;
	      source += rho - fv->rho;
	      source *= wt_func * det_J * h3 * wt;

	      lec->R[peqn][i] += source;
	    }
	}
    }
  /*
   * Jacobian terms_________________________________________________________________
   */

  if ( af->Assemble_Jacobian )
    {

      if( pd->e[pg->imtrx][eqn] )
	{
	  for( i=0; i<ei[pg->imtrx]->dof[eqn]; i++ )
	    {
	      wt_func = bf[eqn]->phi[i];

	      /* J_RHO_RHO */
	      var = DENSITY_EQN;

	      if( pd->v[pg->imtrx][var] )
		{
		  pvar = upd->vp[pg->imtrx][var];

		  for( j=0; j<ei[pg->imtrx]->dof[var]; j++ )
		    {
		      source = 0.0;
		      source -= bf[var]->phi[j];
		      source *= wt_func*det_J*wt*h3;

		      lec->J[peqn][pvar][i][j] += source;

		    }
		}

	      /*
	       * J_RHO_MOM
	       */

	      var = MOMENT1;

	      if( pd->v[pg->imtrx][var] )
		{
		  pvar = upd->vp[pg->imtrx][var];


		  for( j=0; j<ei[pg->imtrx]->dof[var]; j++ )
		    {
		      source = 0;
		      source -= d_rho_dMOM[1][j];
		      source *= wt_func*det_J*wt*h3;

		      lec->J[peqn][pvar][i][j] += source;

		    }
		}

	      /*
	       * J_RHO_T
	       */

	      var = TEMPERATURE;

	      if( pd->v[pg->imtrx][var] )
		{
		  pvar = upd->vp[pg->imtrx][var];

		  for( j=0; j<ei[pg->imtrx]->dof[var]; j++ )
		    {

		      source = 0;
		      source -= d_rho_dT[j];
		      source *= wt_func*det_J*wt*h3;

		      lec->J[peqn][pvar][i][j] += source;

		    }
		}

	      var = MASS_FRACTION;
	      if( pd->v[pg->imtrx][var] )
		{

		  for( a=0 ; a<pd->Num_Species_Eqn; a++ )
		    {

		      pvar = upd->vp[pg->imtrx][var];

		      for( j=0; j<ei[pg->imtrx]->dof[var]; j++ )
			{
			  source = 0;
			  source -= d_rho_dC[a][j];
			  source *= wt_func*det_J*wt*h3;

			  lec->J[peqn][pvar][i][j] += source;

			}
		    }
		}

	      	    /*
		     * J_e_d
		     */
	      for ( b=0; b<dim; b++)
		{
		  var = MESH_DISPLACEMENT1+b;
		  if ( pd->v[pg->imtrx][var] )
		    {
		      pvar = upd->vp[pg->imtrx][var];
		      for ( j=0; j<ei[pg->imtrx]->dof[var]; j++)
			{
			  double dh3dmesh_bj = fv->dh3dq[b] * bf[var]->phi[j];

			  double d_det_J_dmeshbj = bf[eqn]->d_det_J_dm[b][j];


			  source = 0.;
			  source = rho - fv->rho;
			  source *= wt_func*
			    (h3 * d_det_J_dmeshbj +
			     dh3dmesh_bj * det_J)
			    * wt;

			  lec->J[peqn][pvar][i][j] += source;
			}


		    }
		}

	    }
	}
    }
  return(0);
}

double PBEVolumeSource (double time,
			double dt,
			double tt,
			double dFVS_dv[DIM][MDE],
			double dFVS_dT[MDE],
			double dFVS_dx[DIM][MDE],
			double dFVS_dC[MAX_CONC][MDE],
			double dFVS_dMOM[MAX_MOMENTS][MDE])
{
  int a,b,j, dim = pd->Num_Dim, var;
  double source =0.;
  double phi_j;
  int wim;
  int err;

  double vconv[MAX_PDIM]; /*Calculated convection velocity */
  double vconv_old[MAX_PDIM]; /*Calculated convection velocity at previous time*/
  CONVECTION_VELOCITY_DEPENDENCE_STRUCT d_vconv_struct;
  CONVECTION_VELOCITY_DEPENDENCE_STRUCT *d_vconv = &d_vconv_struct;

  DENSITY_DEPENDENCE_STRUCT d_rho_struct;
  DENSITY_DEPENDENCE_STRUCT *d_rho = &d_rho_struct;

  double rho = density(d_rho, time);

  double d_rho_dt = 0;
  double d_rho_dt_dT = 0;
  double d_rho_dt_dMOM1 = 0;
  double d_rho_dt_dC[MAX_CONC];

  double grad_rho[DIM];
  double d_grad_rho_dT[DIM];
  double d_grad_rho_dMOM1[DIM][MDE];
  double d_grad_rho_dC[MAX_CONC][DIM];

  memset( d_rho_dt_dC, 0 , sizeof(double)*MAX_CONC );
  memset( grad_rho, 0 , sizeof(double)*DIM );
  memset( d_grad_rho_dT, 0 , sizeof(double)*DIM );
  memset( d_grad_rho_dMOM1, 0 , sizeof(double)*DIM*MDE );
  memset( d_grad_rho_dC, 0 , sizeof(double)*MAX_CONC*DIM );

  if( (pd->CoordinateSystem == CARTESIAN) ||
      (pd->CoordinateSystem == CYLINDRICAL) )
    {
      wim = pd->Num_Dim;
    }
  else if ( (pd->CoordinateSystem == SWIRLING) ||
	    (pd->CoordinateSystem == PROJECTED_CARTESIAN) )
    {
      wim = 3;
    }
  else
    {
      wim = VIM;
    }

  memset( dFVS_dv, 0 , sizeof(double)*DIM*MDE );
  memset( dFVS_dT, 0 , sizeof(double)*MDE );
  memset( dFVS_dx, 0 , sizeof(double)*DIM*MDE );
  memset( dFVS_dMOM,    0 , sizeof(double)*MAX_MOMENTS*MDE );

  if( cr->MeshMotion == ARBITRARY ||
      cr->MeshMotion == LAGRANGIAN ||
      cr->MeshMotion == DYNAMIC_LAGRANGIAN)
    {
      err = get_convection_velocity(vconv, vconv_old, d_vconv, dt, tt);
      EH(err, "Error in calculating effective convection velocity");
    }
  else if ( cr->MeshMotion == TOTAL_ALE)
    {
      err = get_convection_velocity_rs(vconv, vconv_old, d_vconv, dt, tt);
      EH(err, "Error in calculating effective convection velocity_rs");
    }

  int species_BA_g;
  int species_CO2_g;
  int species_CO2_l;
  int species_BA_l;

  err = get_foam_pbe_indices(NULL, NULL, &species_BA_l, &species_BA_g, &species_CO2_l, &species_CO2_g);
  if (err) return 0;

  double M_BA = mp->u_species_source[species_BA_l][0];
  double M_CO2 = mp->u_species_source[species_CO2_l][0];
  double rho_bubble = 0;
  double rho_foam = mp->u_density[0];
  double ref_press = mp->u_density[1];
  double Rgas_const = mp->u_density[2];

  if (fv->c[species_BA_g] > PBE_FP_SMALL || fv->c[species_CO2_g] > PBE_FP_SMALL) {
    rho_bubble = (ref_press/(Rgas_const*fv->T)) *
      (fv->c[species_CO2_g]*M_CO2 + fv->c[species_BA_g]*M_BA)/(fv->c[species_CO2_g] + fv->c[species_BA_g]);
  }

  double inv_mom_frac = 1/(1 + fv->moment[1]);
  rho = rho_bubble*(fv->moment[1]*inv_mom_frac) + rho_foam*inv_mom_frac;
  d_rho_dt = (rho_bubble - rho_foam)*(fv_dot->moment[1]*inv_mom_frac*inv_mom_frac);
  for (a = 0; a < wim; a++) {
    grad_rho[a] = (rho_bubble - rho_foam)*(fv->grad_moment[1][a]*inv_mom_frac*inv_mom_frac);
  }

  var = TEMPERATURE;
  if (pd->v[pg->imtrx][var] )
    {
      d_rho_dt_dT = (-rho_bubble)/fv->T * (fv_dot->moment[1]*inv_mom_frac*inv_mom_frac);
      for (a = 0; a < wim; a++) {
	d_grad_rho_dT[a] = (-rho_bubble)/fv->T * (fv->grad_moment[1][a]*inv_mom_frac*inv_mom_frac);
      }
    }

  var = MOMENT1;
  if (pd->v[pg->imtrx][var] )
    {

      d_rho_dt_dMOM1 = (rho_bubble - rho_foam) * ((1 + fv->moment[1])*((1 + 2. * tt) / dt) - 2 * fv_dot->moment[1]) *
	inv_mom_frac * inv_mom_frac * inv_mom_frac;
      for ( j=0; j<ei[pg->imtrx]->dof[var]; j++)
	{
	  for (a = 0; a < wim; a++) {
	    d_grad_rho_dMOM1[a][j] = (rho_bubble - rho_foam) * ((1 + fv->moment[1])*(bf[var]->grad_phi[j][a]) - 2 * fv->grad_moment[1][a]) *
	      inv_mom_frac * inv_mom_frac * inv_mom_frac;
	  }
	}
    }

  var = MASS_FRACTION;
  if (pd->v[pg->imtrx][var])
    {
      if (fv->c[species_BA_g] > PBE_FP_SMALL || fv->c[species_CO2_g] > PBE_FP_SMALL) {
	d_rho_dt_dC[species_BA_g] = (fv_dot->moment[1]*inv_mom_frac*inv_mom_frac) * (ref_press/(Rgas_const*fv->T)) *
	  ((M_BA - M_CO2)*fv->c[species_CO2_g])/
	  ((fv->c[species_CO2_g] + fv->c[species_BA_g]) * (fv->c[species_CO2_g] + fv->c[species_BA_g]));

	d_rho_dt_dC[species_CO2_g] = (fv_dot->moment[1]*inv_mom_frac*inv_mom_frac) * (ref_press/(Rgas_const*fv->T)) *
	  ((M_CO2-M_BA)*fv->c[species_BA_g])/
	  ((fv->c[species_CO2_g] + fv->c[species_BA_g]) * (fv->c[species_CO2_g] + fv->c[species_BA_g]));

	for (a = 0; a < wim; a++) {
	  d_grad_rho_dC[species_BA_g][a] = (fv->grad_moment[1][a]*inv_mom_frac*inv_mom_frac) * (ref_press/(Rgas_const*fv->T)) *
	    ((M_BA - M_CO2)*fv->c[species_CO2_g])/
	    ((fv->c[species_CO2_g] + fv->c[species_BA_g]) * (fv->c[species_CO2_g] + fv->c[species_BA_g]));

	  d_grad_rho_dC[species_CO2_g][a] = (fv->grad_moment[1][a]*inv_mom_frac*inv_mom_frac) * (ref_press/(Rgas_const*fv->T)) *
	    ((M_CO2-M_BA)*fv->c[species_BA_g])/
	    ((fv->c[species_CO2_g] + fv->c[species_BA_g]) * (fv->c[species_CO2_g] + fv->c[species_BA_g]));
	}
      } else {
	d_rho_dt_dC[species_BA_g] = 0;
	d_rho_dt_dC[species_CO2_g] = 0;
	for (a = 0; a < wim; a++) {
	  d_grad_rho_dC[species_BA_g][a] = 0;

	  d_grad_rho_dC[species_CO2_g][a] = 0;
	}
      }

    }

  source = 0;
  for (a = 0; a < wim; a++) {
    source += vconv[a] * grad_rho[a];
  }

  source += d_rho_dt;
  source *= (1/rho);

  if (  af->Assemble_Jacobian )
    {
      for (b = 0; b < wim; b++) {
	var = VELOCITY1 + b;

	for( j=0; pd->v[pg->imtrx][var] && j<ei[pg->imtrx]->dof[var]; j++)
	  {
	    dFVS_dv[b][j] = 0.0;
	    for( a=0; a<wim; a++)
	      {
		dFVS_dv[b][j] += d_vconv->v[a][b][j] * grad_rho[a];
	      }

	    dFVS_dv[b][j] *= (1/rho);
	  }
      }

      var = TEMPERATURE;

      for( j=0; pd->v[pg->imtrx][var] && j<ei[pg->imtrx]->dof[var]; j++)
	{
	  dFVS_dT[j] = 0.;
	  phi_j = bf[var]->phi[j];

	  for( a=0; a<wim; a++)
	    {
	      dFVS_dT[j] += d_vconv->T[a][j] * grad_rho[a] + vconv[a] * d_grad_rho_dT[a]*phi_j;
	    }

	  dFVS_dT[j] += d_rho_dt_dT*phi_j;
	  dFVS_dT[j] *= (1/rho);
	  // differentiate 1/rho product rule
	  dFVS_dT[j] += -d_rho->T[j]*(1/rho)*source;
	}

      for( b=0; b < dim; b++)
	{
	  var = MESH_DISPLACEMENT1 + b;

	  for( j=0; pd->v[pg->imtrx][var] && j<ei[pg->imtrx]->dof[var]; j++)
	    {
	      dFVS_dx[b][j] = 0.0;
	      for( a=0; a<wim; a++)
		{
		  dFVS_dx[b][j] += d_vconv->X[a][b][j] * grad_rho[a] +
		    (rho_bubble - rho_foam)*(fv->d_grad_moment_dmesh[1][a][b][j]*inv_mom_frac*inv_mom_frac);
		}

	      dFVS_dx[b][j] *= (1/rho);
	    }
	}

      var = MOMENT1;

      for( j=0; pd->v[pg->imtrx][var] && j<ei[pg->imtrx]->dof[var]; j++)
	{
	  dFVS_dMOM[1][j] = 0.0;
	  phi_j = bf[var]->phi[j];

	  for( a=0; a<wim; a++)
	    {
	      dFVS_dMOM[1][j] += vconv[a] * d_grad_rho_dMOM1[a][j];
	    }

	  dFVS_dMOM[1][j] += d_rho_dt_dMOM1 * phi_j;
	  dFVS_dMOM[1][j] *= (1/rho);

	  // differentiate 1/rho product rule
	  dFVS_dMOM[1][j] += -d_rho->moment[1][j]*(1/rho)*source;

	}


      var = MASS_FRACTION;

      for( j=0; pd->v[pg->imtrx][var] && j <ei[pg->imtrx]->dof[var]; j++ )
	{
	  dFVS_dC[species_BA_g][j] = 0.0;
	  dFVS_dC[species_CO2_g][j] = 0.0;
	  phi_j = bf[var]->phi[j];
	  for( a=0; a<wim; a++)
	    {
	      dFVS_dC[species_BA_g][j] += d_vconv->C[a][species_BA_g][j] * grad_rho[a] + vconv[a]*d_grad_rho_dC[species_BA_g][a]*phi_j;
	      dFVS_dC[species_CO2_g][j] += d_vconv->C[a][species_CO2_g][j] * grad_rho[a] + vconv[a]*d_grad_rho_dC[species_CO2_g][a]*phi_j;
	    }

	  dFVS_dC[species_BA_g][j] += d_rho_dt_dC[species_BA_g]*phi_j;
	  dFVS_dC[species_CO2_g][j] += d_rho_dt_dC[species_CO2_g]*phi_j;
	  dFVS_dC[species_BA_g][j] *= (1/rho);
	  dFVS_dC[species_CO2_g][j] *= (1/rho);

	  // differentiate 1/rho product rule
	  dFVS_dC[species_BA_g][j] += -d_rho->C[species_BA_g][j]*(1/rho)*source;
	  dFVS_dC[species_CO2_g][j] += -d_rho->C[species_CO2_g][j]*(1/rho)*source;
	}
    }
  return ( source );
}

double PBEVolumeSource_rhoeqn(double time,
			      double dt,
			      double tt,
			      double dFVS_drho[MDE])
{
  int a,j,var;
  double source =0.;
  double phi_j;
  int wim;
  int err;

  double vconv[MAX_PDIM]; /*Calculated convection velocity */
  double vconv_old[MAX_PDIM]; /*Calculated convection velocity at previous time*/
  CONVECTION_VELOCITY_DEPENDENCE_STRUCT d_vconv_struct;
  CONVECTION_VELOCITY_DEPENDENCE_STRUCT *d_vconv = &d_vconv_struct;

  if( (pd->CoordinateSystem == CARTESIAN) ||
      (pd->CoordinateSystem == CYLINDRICAL) )
    {
      wim = pd->Num_Dim;
    }
  else if ( (pd->CoordinateSystem == SWIRLING) ||
	    (pd->CoordinateSystem == PROJECTED_CARTESIAN) )
    {
      wim = 3;
    }
  else
    {
      wim = VIM;
    }

  memset( dFVS_drho, 0 , sizeof(double)*MDE );

  if( cr->MeshMotion == ARBITRARY ||
      cr->MeshMotion == LAGRANGIAN ||
      cr->MeshMotion == DYNAMIC_LAGRANGIAN)
    {
      err = get_convection_velocity(vconv, vconv_old, d_vconv, dt, tt);
      EH(err, "Error in calculating effective convection velocity");
    }
  else if ( cr->MeshMotion == TOTAL_ALE)
    {
      err = get_convection_velocity_rs(vconv, vconv_old, d_vconv, dt, tt);
      EH(err, "Error in calculating effective convection velocity_rs");
    }


  source = 0;

  if (fv->rho != 0) {
    for (a = 0; a < wim; a++) {
      source += vconv[a] * fv->grad_rho[a];
    }

    source += fv_dot->rho;
    source *= (1/fv->rho);
  } else {
    source = 0;
    return source;
  }


  if (  af->Assemble_Jacobian )
    {
      var = DENSITY_EQN;

      for( j=0; pd->v[pg->imtrx][var] && j<ei[pg->imtrx]->dof[var]; j++)
	{
	  dFVS_drho[j] = 0.0;
	  phi_j = bf[var]->phi[j];

	  for( a=0; a<wim; a++)
	    {
	      dFVS_drho[j] += vconv[a] * bf[var]->grad_phi[j][a];
	    }

	  dFVS_drho[j] +=  (1 + 2. * tt) * phi_j / dt;
	  dFVS_drho[j] *= (1/fv->rho);

	  // differentiate 1/rho product rule
	  dFVS_drho[j] += -phi_j*(1/fv->rho)*source;

	}

    }
  return ( source );
}


int
assemble_moments(double time,	/* present time value */
		 double tt,	/* parameter to vary time integration from explicit (tt = 1) to implicit (tt = 0) */
		 double dt,	/* current time step size */
		 const PG_DATA *pg_data )
{
  int eqn, var, peqn, pvar, dim, p, b, qq, w, i, j, status;

  dbl mass;		         	/* For terms and their derivatives */

  dbl advection;			/* For terms and their derivatives */
  dbl advection_a;
  dbl advection_b;
  dbl advection_c;
  dbl advection_d;
  dbl advection_e;

  dbl source;
  dbl diffusion;

  /*
   * Galerkin weighting functions for i-th energy residuals
   * and some of their derivatives...
   */

  dbl phi_i;
  dbl grad_phi_i[DIM];

  /*
   * Petrov-Galerkin weighting functions for i-th residuals
   * and some of their derivatives...
   */

  dbl wt_func;

/* SUPG variables */
  dbl h_elem=0, h_elem_inv=0, h_elem_deriv=0, h_elem_inv_deriv=0.;
  dbl supg, d_wt_func;

  /*
   * Interpolation functions for variables and some of their derivatives.
   */

  dbl phi_j;
  dbl grad_phi_j[DIM];

  dbl h3;			/* Volume element (scale factors). */
  dbl dh3dmesh_bj;		/* Sensitivity to (b,j) mesh dof. */

  dbl det_J;

  dbl d_det_J_dmeshbj;			/* for specified (b,j) mesh dof */
  dbl wt;

  double vconv[MAX_PDIM]; /*Calculated convection velocity */
  double vconv_old[MAX_PDIM]; /*Calculated convection velocity at previous time*/
  CONVECTION_VELOCITY_DEPENDENCE_STRUCT d_vconv_struct;
  CONVECTION_VELOCITY_DEPENDENCE_STRUCT *d_vconv = &d_vconv_struct;


  double msource[MAX_MOMENTS];
  MOMENT_SOURCE_DEPENDENCE_STRUCT *d_msource;


  int err;

  int mom;

  const double *hsquared = pg_data->hsquared ;
  const double *vcent = pg_data->v_avg; /* Average element velocity, which is the
	  centroid velocity for Q2 and the average
	  of the vertices for Q1. It comes from
	  the routine "element_velocity." */



  /* initialize grad_phi_j */
  for (i = 0; i < DIM; i++) {
    grad_phi_j[i] = 0;
  }

  /*   static char yo[] = "assemble_energy";*/

  status = 0;

  /*
   * Unpack variables from structures for local convenience...
   */

  dim   = pd->Num_Dim;

  eqn   = R_MOMENT0;

  /*
   * Bail out fast if there's nothing to do...
   */

  if ( ! pd->e[pg->imtrx][eqn] )
    {
      return(status);
    }

  d_msource = calloc(sizeof(MOMENT_SOURCE_DEPENDENCE_STRUCT), 1);
  moment_source(msource, d_msource);

  wt = fv->wt;				/* Gauss point weight. */

  h3 = fv->h3;			/* Differential volume element. */

  det_J = bf[eqn]->detJ;		/* Really, ought to be mesh eqn. */

  supg = 0.;


  if(supg!=0.)
    {
      h_elem = 0.;
      for ( p=0; p<dim; p++)
	{
	  h_elem += vcent[p] * vcent[p] / hsquared[p];
	}
      h_elem = sqrt(h_elem)/2.;
      if(h_elem == 0.)
	{
	  h_elem_inv=0.;
	}
      else
	{
	  h_elem_inv=1./h_elem;
	}

    }
/* end Petrov-Galerkin addition */

/* get the convection velocity (it's different for arbitrary and
   lagrangian meshes) */
  if( cr->MeshMotion == ARBITRARY ||
      cr->MeshMotion == LAGRANGIAN ||
      cr->MeshMotion == DYNAMIC_LAGRANGIAN)
    {
      err = get_convection_velocity(vconv, vconv_old, d_vconv, dt, tt);
      EH(err, "Error in calculating effective convection velocity");
    }
  else if ( cr->MeshMotion == TOTAL_ALE)
    {
      err = get_convection_velocity_rs(vconv, vconv_old, d_vconv, dt, tt);
      EH(err, "Error in calculating effective convection velocity_rs");
    }


  /*
   * Residuals___________________________________________________________
   */

  if ( af->Assemble_Residual )
    {
      for (mom = 0; mom < MAX_MOMENTS; mom++) {
	eqn = R_MOMENT0 + mom;
	peqn = upd->ep[pg->imtrx][eqn];
	var = MOMENT0 + mom;
	for ( i=0; i<ei[pg->imtrx]->dof[eqn]; i++)
	  {

#if 1
	    /* this is an optimization for xfem */
	    if ( xfem != NULL )
	      {
		int xfem_active, extended_dof, base_interp, base_dof;
		xfem_dof_state( i, pd->i[pg->imtrx][eqn], ei[pg->imtrx]->ielem_shape,
				&xfem_active, &extended_dof, &base_interp, &base_dof );
		if ( extended_dof && !xfem_active ) continue;
	      }
#endif
	    phi_i = bf[eqn]->phi[i];

	    mass = 0.;
	    if ( pd->TimeIntegration != STEADY )
	      {
		if ( pd->e[pg->imtrx][eqn] & T_MASS )
		  {
		    mass  = fv_dot->moment[mom];
		    mass *= - phi_i * det_J * wt;
		    mass *= h3;
		    mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
		  }
	      }

	    /* only use Petrov Galerkin on advective term - if required */
	    wt_func = bf[eqn]->phi[i];
	    /* add Petrov-Galerkin terms as necessary */
	    if(supg!=0.)
	      {
		for(p=0; p<dim; p++)
		  {
		    wt_func += supg * h_elem_inv * vconv[p] * bf[eqn]->grad_phi[i][p];
		  }
	      }

	    advection = 0.;
	    if ( pd->e[pg->imtrx][eqn] & T_ADVECTION )
	      {

		for ( p=0; p<VIM; p++)
		  {
		    advection += vconv[p]     * fv->grad_moment[mom][p];
		  }

		advection *= - wt_func * det_J * wt;
		advection *= h3;
		advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
	      }

	    diffusion = 0.;
	    if ( pd->e[pg->imtrx][eqn] & T_DIFFUSION )
	      {

		for ( p=0; p<VIM; p++)
		  {
		    diffusion += bf[eqn]->grad_phi[i][p] * fv->grad_moment[mom][p];
		  }

		diffusion *= det_J * wt;
		diffusion *= h3;
		diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
	      }

	    source = 0.;
	    if ( pd->e[pg->imtrx][eqn] & T_SOURCE )
	      {
		source += phi_i * msource[mom] * det_J * wt;
		source *= h3;
		source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
	      }

	    lec->R[peqn][i] +=
	      mass + advection + source + diffusion;

	  }
      }
    }


  /*
   * Jacobian terms...
   */

  if ( af->Assemble_Jacobian )
    {
      for (mom = 0; mom < MAX_MOMENTS; mom++) {
	eqn   = R_MOMENT0 + mom;
	peqn = upd->ep[pg->imtrx][eqn];
	for ( i=0; i<ei[pg->imtrx]->dof[eqn]; i++)
	  {
#if 1
	    /* this is an optimization for xfem */
	    if ( xfem != NULL )
	      {
		int xfem_active, extended_dof, base_interp, base_dof;
		xfem_dof_state( i, pd->i[pg->imtrx][eqn], ei[pg->imtrx]->ielem_shape,
				&xfem_active, &extended_dof, &base_interp, &base_dof );
		if ( extended_dof && !xfem_active ) continue;
	      }
#endif
	    phi_i = bf[eqn]->phi[i];

	    wt_func = bf[eqn]->phi[i];
	    /* add Petrov-Galerkin terms as necessary */
	    if(supg!=0.)
	      {
		for(p=0; p<dim; p++)
		  {
		    wt_func += supg * h_elem_inv * vconv[p] * bf[eqn]->grad_phi[i][p];
		  }
	      }

	    /*
	     * Set up some preliminaries that are needed for the (a,i)
	     * equation for bunches of (b,j) column variables...
	     */

	    for ( p=0; p<VIM; p++)
	      {
		grad_phi_i[p] = bf[eqn]->grad_phi[i][p];
	      }

	    /*
	     * J_e_T
	     */
	    var = TEMPERATURE;
	    if ( pd->v[pg->imtrx][var] )
	      {
		pvar = upd->vp[pg->imtrx][var];
		for ( j=0; j<ei[pg->imtrx]->dof[var]; j++)
		  {
		    phi_j = bf[var]->phi[j];

		    source = 0.;
		    if ( pd->e[pg->imtrx][eqn] & T_SOURCE )
		      {
			source += phi_i * d_msource->T[mom][j] * det_J * wt;
			source *= h3;
			source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
		      }

		    lec->J[peqn][pvar][i][j] += source;
		  }
	      }

	    /*
	     * J_e_MOM
	     */
	    for (b = 0; b < MAX_MOMENTS; b++) {
	      var = MOMENT0 + b;
	      if ( pd->v[pg->imtrx][var] )
		{
		  pvar = upd->vp[pg->imtrx][var];
		  for ( j=0; j<ei[pg->imtrx]->dof[var]; j++)
		    {
		      phi_j = bf[var]->phi[j];

		      for ( p=0; p<VIM; p++)
			{
			  grad_phi_j[p] = bf[var]->grad_phi[j][p];
			}

		      mass = 0.;
		      if ( pd->TimeIntegration != STEADY )
			{
			  if (mom == b) {
			    if ( pd->e[pg->imtrx][eqn] & T_MASS )
			      {

				mass  = (1 + 2. * tt) * phi_j / dt;

				mass *= - phi_i * det_J * wt;
				mass *= h3;
				mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
			      }
			  }
			}

		      advection = 0.;
		      if ( pd->e[pg->imtrx][eqn] & T_ADVECTION)
			{
			  if (mom == b) {
			    for ( p=0; p<VIM; p++)
			      {
				advection +=
				  vconv[p] * grad_phi_j[p];
			      }
			    advection *= - wt_func * det_J * wt;
			    advection *= h3;
			    advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
			  }
			}

		      diffusion = 0.;
		      if ( pd->e[pg->imtrx][eqn] & T_DIFFUSION )
			{

			  if (mom == b) {
			    for ( p=0; p<VIM; p++)
			      {
				diffusion += bf[eqn]->grad_phi[i][p] * grad_phi_j[p];
			      }

			    diffusion *= det_J * wt;
			    diffusion *= h3;
			    diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
			  }
			}

		      source = 0.;
		      if ( pd->e[pg->imtrx][eqn] & T_SOURCE )
			{
			  //source += phi_i * d_msource->moment[b][j] * det_J * wt;
			  source *= h3;
			  source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
			}

		      lec->J[peqn][pvar][i][j] += mass + advection + source + diffusion;
		    }
		}
	    }


	    /*
	     * J_e_v
	     */
	    for ( b=0; b<VIM; b++)
	      {
		var = VELOCITY1+b;
		if ( pd->v[pg->imtrx][var] )
		  {
		    pvar = upd->vp[pg->imtrx][var];
		    for ( j=0; j<ei[pg->imtrx]->dof[var]; j++)
		      {
			phi_j = bf[var]->phi[j];

			mass = 0.;

			advection = 0.;
			advection_a = 0.;
			if ( pd->e[pg->imtrx][eqn] & T_ADVECTION )
			  {
			    advection_a += wt_func * d_vconv->v[b][b][j] * fv->grad_moment[mom][b];
			    advection_a *= - det_J * wt;
			    advection_a *= h3;
			    advection_a *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
			    if(supg!=0.)
			      {
				h_elem_deriv = 0.;
				if(hsquared[b] != 0.)
				  {
				    h_elem_deriv =
				      vcent[b] * pg_data->dv_dnode[b][j] * h_elem_inv/4./hsquared[b];
				  }
				if (h_elem != 0.) h_elem_inv_deriv = -h_elem_deriv/h_elem/h_elem;

			      }
			    advection_b = 0.;
			    if(supg!=0.)
			      {
				d_wt_func = supg * h_elem_inv * d_vconv->v[b][b][j] * grad_phi_i[b]
				  + supg * h_elem_inv_deriv * vconv[b] * grad_phi_i[b];

				for(p=0;p<dim;p++)
				  {
				    advection_b +=  vconv[p] * fv->grad_moment[mom][p];
				  }

				advection_b *=  d_wt_func;
				advection_b *= - det_J * wt;
				advection_b *= h3;
				advection_b *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
			      }
			    advection = advection_a + advection_b;
			  }

			source = 0.;

			if ( pd->e[pg->imtrx][eqn] & T_SOURCE )
			  {
			    //source += phi_i * d_msource->v[b][j] * det_J * wt;
			    source *= h3;
			    source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
			  }

			lec->J[peqn][pvar][i][j] += mass + advection + source;

		      }
		  }
	      }

	    /*
	     * J_e_d
	     */
	    for ( b=0; b<dim; b++)
	      {
		var = MESH_DISPLACEMENT1+b;
		if ( pd->v[pg->imtrx][var] )
		  {
		    pvar = upd->vp[pg->imtrx][var];
		    for ( j=0; j<ei[pg->imtrx]->dof[var]; j++)
		      {
			phi_j = bf[var]->phi[j];

			dh3dmesh_bj = fv->dh3dq[b] * bf[var]->phi[j];

			d_det_J_dmeshbj = bf[eqn]->d_det_J_dm[b][j];


			if(supg!=0.)
			  {
			    h_elem_deriv = 0.;
			    h_elem_inv_deriv=0.;
			    for( qq=0; qq<dim; qq++ )
			      {
				if(pg_data->hhv[qq][b] != 0.)
				  {
				    h_elem_deriv -=
				      vcent[qq] * vcent[qq] *
				      pg_data->dhv_dxnode[qq][j] * pg_data->hhv[qq][b] * h_elem_inv/4./hsquared[qq]/hsquared[qq];
				  }
			      }
			    if (h_elem != 0.) h_elem_inv_deriv = -h_elem_deriv/h_elem/h_elem;
			    // h_elem_inv_deriv = 0.; /* PRS: NOT SURE WHY THIS IS NOT RIGHT, SO SET TO ZERO */

			  }

			mass = 0.;
			if ( pd->TimeIntegration != STEADY )
			  {
			    if ( pd->e[pg->imtrx][eqn] & T_MASS )
			      {
				mass  = fv_dot->moment[mom];
				mass *= - phi_i *
				  (h3 * d_det_J_dmeshbj +
				   dh3dmesh_bj * det_J)
				  * wt;
				mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
			      }
			  }


			advection = 0.;

			if ( pd->e[pg->imtrx][eqn] & T_ADVECTION )
			  {
			    /*
			     * Four parts:
			     *	Here, by "v" we mean "vconv" which is
			     *    vconv = v - vmesh,   vmesh ~ xdot ~ dd/dt
			     *
			     *	d/dmesh [ Int(... vconv.grad(T) h3 detJ ) ]
			     *
			     *	(a)	Int( Cp * vconv.d(grad(T))/dmesh h3 detJ )
			     *	(b)	Int( Cp * vconv.grad(T) h3 ddetJ/dmesh )
			     *	(c)	Int( Cp * dvconv/dmesh . grad(T) h3 detJ )
			     *	(d)	Int( Cp * vconv.grad(T) dh3/dmesh detJ )
			     *	(e)	Int( dCp/dmesh * vconv.grad(T) h3 detJ )
			     *
			     */

			    advection_a = 0.;
			    for ( p=0; p<dim; p++)
			      {
				advection_a += vconv[p] *
				  fv->d_grad_moment_dmesh[mom][p] [b][j];
			      }
			    advection_a *= -wt_func * h3 * det_J * wt;

			    advection_b = 0.;
			    for ( p=0; p<dim; p++)
			      {
				advection_b += vconv[p] * fv->grad_moment[mom][p];
			      }
			    advection_b *= - wt_func *  h3 *
			      d_det_J_dmeshbj *wt;

			    advection_c = 0.;
			    if ( pd->TimeIntegration != STEADY )
			      {
				if ( pd->e[pg->imtrx][eqn] & T_MASS )
				  {
				    for ( p=0; p<dim; p++)
				      {
					advection_c +=
					  d_vconv->X[p][b][j] * fv->grad_moment[mom][p];
				      }
				    advection_c *= -wt_func * h3 *
				      det_J * wt;
				  }
			      }

			    advection_d = 0.;
			    for ( p=0; p<dim; p++)
			      {
				advection_d += vconv[p] * fv->grad_moment[mom][p];
			      }
			    advection_d *= - wt_func * dh3dmesh_bj *
			      det_J *wt;

			    advection_e = 0.;
			    if(supg != 0.)
			      {
				d_wt_func = 0.;
				for( p=0; p<dim; p++ )
				  {
				    d_wt_func += supg
				      * (h_elem_inv *fv->v[p]* bf[eqn]->d_grad_phi_dmesh[i][p] [b][j]
					 +  h_elem_inv_deriv * fv->v[p] *grad_phi_i[p]);


				    advection_e += vconv[p] * fv->grad_moment[mom][p];
				  }
				advection_e *= -d_wt_func * h3 * det_J * wt;
			      }

			    advection = advection_a + advection_b + advection_c
			      + advection_d + advection_e;

			    advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
			  }


			source = 0.;

			if ( pd->e[pg->imtrx][eqn] & T_SOURCE )
			  {
			    source = phi_i * (msource[mom] *          d_det_J_dmeshbj * h3 +
					      msource[mom] *          det_J           * dh3dmesh_bj) * wt;
					      //d_msource->X[b][j]*  det_J           * h3) * wt;

			    source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
			  }

			lec->J[peqn][pvar][i][j] += mass + advection + source;
		      }
		  }
	      }

	    /*
	     * J_e_c
	     */
	    var = MASS_FRACTION;
	    if ( pd->e[pg->imtrx][eqn] && pd->v[pg->imtrx][var] )
	      {
		for ( w=0; w<pd->Num_Species_Eqn; w++)
		  {
		    for ( j=0; j<ei[pg->imtrx]->dof[var]; j++)
		      {
			phi_j = bf[var]->phi[j];

			source = 0.;
			if ( pd->e[pg->imtrx][eqn] & T_SOURCE )
			  {
			    source += phi_i * d_msource->C[mom][w][j] * det_J * wt;
			    source *= h3;
			    source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
			  }

			lec->J[peqn][MAX_PROB_VAR + w][i][j] += source;
		      }
		  }
	      }

	  }
      }
    }

  free(d_msource);
  return(status);
} /* end of assemble_moments */
