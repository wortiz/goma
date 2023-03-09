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
#include "mm_fill_stress.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "el_elm.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "mm_eh.h"
#include "mm_fill_ls.h"
#include "mm_fill_stabilization.h"
#include "mm_fill_stress_legacy.h"
#include "mm_fill_stress_log_conf.h"
#include "mm_mp.h"
#include "mm_mp_const.h"
#include "mm_mp_structs.h"
#include "mm_viscosity.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "std.h"

bool is_evss_f_model(int model) {
  switch (model) {
  case EVSS_F:
  case EVSS_GRADV:
  case CONF:
  case LOG_CONF_GRADV:
  case LOG_CONF:
  case SQRT_CONF:
  case LOG_CONF_TRANSIENT:
  case LOG_CONF_TRANSIENT_GRADV:
    return true;
  default:
    return false;
  }
}

void ve_stress_term(dbl mu,
                    dbl mus,
                    VISCOSITY_DEPENDENCE_STRUCT *d_mu,
                    VISCOSITY_DEPENDENCE_STRUCT *d_mus,
                    dbl stress[DIM][DIM],
                    STRESS_DEPENDENCE_STRUCT *d_stress) {
  bool compute_derivatives = (d_stress != NULL);
  memset(stress, 0, sizeof(dbl) * DIM * DIM);
  switch (vn->evssModel) {
  case EVSS_F:
  case EVSS_GRADV: {
    for (int mode = 0; mode < vn->modes; mode++) {
      for (int i = 0; i < VIM; i++) {
        for (int j = 0; j < VIM; j++) {
          stress[i][j] += fv->S[mode][i][j];
        }
      }
    }
  } break;
  case CONF: {
    // conformation tensor
    for (int mode = 0; mode < vn->modes; mode++) {
      /* get polymer viscosity */
      dbl mup = mu - mus;
      dbl lambda = ve[mode]->time_const;
      for (int ii = 0; ii < VIM; ii++) {
        for (int jj = 0; jj < VIM; jj++) {
          stress[ii][jj] += -(mup / lambda) * (delta(ii, jj) - fv->S[mode][ii][jj]);
        }
      }
    }
  } break;
  case LOG_CONF_GRADV:
  case LOG_CONF: {
    // conformation tensor
    dbl exp_s[DIM][DIM];
    dbl eig_values[DIM];
    dbl R1[DIM][DIM];
    for (int mode = 0; mode < vn->modes; mode++) {
      compute_exp_s(fv->S[mode], exp_s, eig_values, R1);
      /* get polymer viscosity */
      dbl mup = mu - mus;
      dbl lambda = ve[mode]->time_const;
      for (int ii = 0; ii < VIM; ii++) {
        for (int jj = 0; jj < VIM; jj++) {
          stress[ii][jj] += -(mup / lambda) * (delta(ii, jj) - exp_s[ii][jj]);
        }
      }
    }
  } break;
  case LOG_CONF_TRANSIENT:
  case LOG_CONF_TRANSIENT_GRADV: {
    // conformation tensor
    dbl exp_s[DIM][DIM];
    dbl eig_values[DIM];
    dbl R1[DIM][DIM];
    for (int mode = 0; mode < vn->modes; mode++) {
      if (pg->imtrx == upd->matrix_index[POLYMER_STRESS11]) {
        compute_exp_s(fv_old->S[mode], exp_s, eig_values, R1);
      } else {
        compute_exp_s(fv->S[mode], exp_s, eig_values, R1);
      }
      /* get polymer viscosity */
      dbl mup = mu - mus;
      dbl lambda = ve[mode]->time_const;
      for (int ii = 0; ii < VIM; ii++) {
        for (int jj = 0; jj < VIM; jj++) {
          stress[ii][jj] += -(mup / lambda) * (delta(ii, jj) - exp_s[ii][jj]);
        }
      }
    }
  } break;
  case SQRT_CONF: {
    for (int mode = 0; mode < vn->modes; mode++) {
      /* get polymer viscosity */
      dbl mup = mu - mus;
      dbl lambda = ve[mode]->time_const;

      dbl bdotb[DIM][DIM];
      dbl b[DIM][DIM];
      for (int ii = 0; ii < VIM; ii++) {
        for (int jj = 0; jj < VIM; jj++) {
          if (ii <= jj) {
            b[ii][jj] = fv->S[mode][ii][jj];
            b[jj][ii] = b[ii][jj];
          }
        }
      }

      tensor_dot(b, b, bdotb, VIM);

      for (int ii = 0; ii < VIM; ii++) {
        for (int jj = 0; jj < VIM; jj++) {
          stress[ii][jj] += -(mup / lambda) * (delta(ii, jj) - bdotb[ii][jj]);
        }
      }
    }
  } break;
  default:
    GOMA_EH(GOMA_ERROR, "Unkown viscoelastic model %d", vn->evssModel);
  }

  // Jacobian portion
  if (compute_derivatives) {
    int v_s[MAX_MODES][DIM][DIM];
    (void)stress_eqn_pointer(v_s);
    switch (vn->evssModel) {
    case EVSS_F:
    case EVSS_GRADV: {
      for (int mode = 0; mode < vn->modes; mode++) {
        for (int i = 0; i < VIM; i++) {
          for (int j = 0; j < VIM; j++) {
            for (int p = 0; p < VIM; p++) {
              for (int q = 0; q < VIM; q++) {
                int var = v_s[mode][p][q];
                for (int k = 0; k < ei[pg->imtrx]->dof[var]; k++) {
                  d_stress->S[i][j][mode][p][q][k] +=
                      ((double)delta(i, p) * (double)delta(j, q)) * bf[var]->phi[k];
                }
              }
            }
          }
        }
      }
    } break;
    case CONF: {
      // conformation tensor
      for (int mode = 0; mode < vn->modes; mode++) {
        /* get polymer viscosity */
        dbl mup = mu - mus;
        dbl lambda = ve[mode]->time_const;
        // stress
        for (int i = 0; i < VIM; i++) {
          for (int j = 0; j < VIM; j++) {
            for (int p = 0; p < VIM; p++) {
              for (int q = 0; q < VIM; q++) {
                int var = v_s[mode][p][q];
                for (int k = 0; k < ei[pg->imtrx]->dof[var]; k++) {
                  d_stress->S[i][j][mode][p][q][k] += -(mup / lambda) *
                                                      -((double)delta(i, p) * (double)delta(j, q)) *
                                                      bf[var]->phi[k];
                }
              }
            }
          }
        }

        // mesh
        int var = MESH_DISPLACEMENT1;
        if (pd->v[pg->imtrx][var]) {
          for (int a = 0; a < pd->Num_Dim; a++) {
            for (int k = 0; k < ei[pg->imtrx]->dof[var + a]; k++) {
              for (int ii = 0; ii < VIM; ii++) {
                for (int jj = 0; jj < VIM; jj++) {
                  stress[ii][jj] += -((d_mu->X[a][k] - d_mus->X[a][k]) / lambda) *
                                    (delta(ii, jj) - fv->S[mode][ii][jj]);
                }
              }
            }
          }
        }

        // velocity
        var = VELOCITY1;
        if (pd->v[pg->imtrx][var]) {
          for (int a = 0; a < WIM; a++) {
            for (int k = 0; k < ei[pg->imtrx]->dof[var + a]; k++) {
              for (int ii = 0; ii < VIM; ii++) {
                for (int jj = 0; jj < VIM; jj++) {
                  stress[ii][jj] += -((d_mu->v[a][k] - d_mus->v[a][k]) / lambda) *
                                    (delta(ii, jj) - fv->S[mode][ii][jj]);
                }
              }
            }
          }
        }

        // temperature
        var = TEMPERATURE;
        if (pd->v[pg->imtrx][var]) {
          for (int k = 0; k < ei[pg->imtrx]->dof[var]; k++) {
            for (int ii = 0; ii < VIM; ii++) {
              for (int jj = 0; jj < VIM; jj++) {
                stress[ii][jj] +=
                    -((d_mu->T[k] - d_mus->T[k]) / lambda) * (delta(ii, jj) - fv->S[mode][ii][jj]);
              }
            }
          }
        }

        // fill
        var = FILL;
        if (pd->v[pg->imtrx][var]) {
          for (int k = 0; k < ei[pg->imtrx]->dof[var]; k++) {
            for (int ii = 0; ii < VIM; ii++) {
              for (int jj = 0; jj < VIM; jj++) {
                stress[ii][jj] +=
                    -((d_mu->F[k] - d_mus->F[k]) / lambda) * (delta(ii, jj) - fv->S[mode][ii][jj]);
              }
            }
          }
        }
      } // mode loop
    } break;
    case LOG_CONF_GRADV:
    case LOG_CONF: {
      // conformation tensor
      dbl exp_s[DIM][DIM];
      dbl eig_values[DIM];
      dbl R1[DIM][DIM];
      for (int mode = 0; mode < vn->modes; mode++) {
        compute_exp_s(fv->S[mode], exp_s, eig_values, R1);
        /* get polymer viscosity */
        dbl mup = mu - mus;
        dbl lambda = ve[mode]->time_const;
        // stress
        int var;
        for (int p = 0; p < VIM; p++) {
          for (int q = 0; q < VIM; q++) {
            for (int r = 0; r < VIM; r++) {
              for (int c = 0; c < VIM; c++) {
                var = v_s[mode][r][c];
                dbl s[DIM][DIM];
                for (int ii = 0; ii < VIM; ii++) {
                  for (int jj = 0; jj < VIM; jj++) {
                    s[ii][jj] = fv->S[mode][ii][jj];
                  }
                }

                dbl exp_s_p[DIM][DIM];
                dbl eig_values_p[DIM];
                dbl s_p = (1e-8) * s[r][c];
                s[r][c] += s_p;

                compute_exp_s(s, exp_s_p, eig_values_p, R1);

                for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                  d_stress->S[p][q][mode][r][c][j] +=
                      -(mup / lambda) * ((exp_s[p][q] - exp_s_p[p][q]) / s_p) * bf[var]->phi[j];
                }
              }
            }
          }
        }

        // mesh
        var = MESH_DISPLACEMENT1;
        if (pd->v[pg->imtrx][var]) {
          for (int a = 0; a < pd->Num_Dim; a++) {
            for (int k = 0; k < ei[pg->imtrx]->dof[var + a]; k++) {
              for (int ii = 0; ii < VIM; ii++) {
                for (int jj = 0; jj < VIM; jj++) {
                  stress[ii][jj] += -((d_mu->X[a][k] - d_mus->X[a][k]) / lambda) *
                                    (delta(ii, jj) - exp_s[ii][jj]);
                }
              }
            }
          }
        }

        // velocity
        var = VELOCITY1;
        if (pd->v[pg->imtrx][var]) {
          for (int a = 0; a < WIM; a++) {
            for (int k = 0; k < ei[pg->imtrx]->dof[var + a]; k++) {
              for (int ii = 0; ii < VIM; ii++) {
                for (int jj = 0; jj < VIM; jj++) {
                  stress[ii][jj] += -((d_mu->v[a][k] - d_mus->v[a][k]) / lambda) *
                                    (delta(ii, jj) - exp_s[ii][jj]);
                }
              }
            }
          }
        }

        // temperature
        var = TEMPERATURE;
        if (pd->v[pg->imtrx][var]) {
          for (int k = 0; k < ei[pg->imtrx]->dof[var]; k++) {
            for (int ii = 0; ii < VIM; ii++) {
              for (int jj = 0; jj < VIM; jj++) {
                stress[ii][jj] +=
                    -((d_mu->T[k] - d_mus->T[k]) / lambda) * (delta(ii, jj) - exp_s[ii][jj]);
              }
            }
          }
        }

        // fill
        var = FILL;
        if (pd->v[pg->imtrx][var]) {
          for (int k = 0; k < ei[pg->imtrx]->dof[var]; k++) {
            for (int ii = 0; ii < VIM; ii++) {
              for (int jj = 0; jj < VIM; jj++) {
                stress[ii][jj] +=
                    -((d_mu->F[k] - d_mus->F[k]) / lambda) * (delta(ii, jj) - exp_s[ii][jj]);
              }
            }
          }
        }
      }
    } break;
    case LOG_CONF_TRANSIENT:
    case LOG_CONF_TRANSIENT_GRADV: {
      // conformation tensor
      dbl exp_s[DIM][DIM];
      dbl eig_values[DIM];
      dbl R1[DIM][DIM];
      for (int mode = 0; mode < vn->modes; mode++) {
        if (pg->imtrx == upd->matrix_index[POLYMER_STRESS11]) {
          compute_exp_s(fv_old->S[mode], exp_s, eig_values, R1);
        } else {
          compute_exp_s(fv->S[mode], exp_s, eig_values, R1);
        }
      }
    } break;
    case SQRT_CONF: {
      for (int mode = 0; mode < vn->modes; mode++) {
        /* get polymer viscosity */
        dbl mup = mu - mus;
        dbl lambda = ve[mode]->time_const;

        dbl bdotb[DIM][DIM];
        dbl b[DIM][DIM];
        for (int ii = 0; ii < VIM; ii++) {
          for (int jj = 0; jj < VIM; jj++) {
            if (ii <= jj) {
              b[ii][jj] = fv->S[mode][ii][jj];
              b[jj][ii] = b[ii][jj];
            }
          }
        }

        tensor_dot(b, b, bdotb, VIM);

        // stress
        int var;
        for (int p = 0; p < VIM; p++) {
          for (int q = 0; q < VIM; q++) {
            for (int r = 0; r < VIM; r++) {
              for (int c = 0; c < VIM; c++) {
                var = v_s[mode][r][c];
                dbl conf[DIM][DIM];
                for (int ii = 0; ii < VIM; ii++) {
                  for (int jj = 0; jj < VIM; jj++) {
                    conf[ii][jj] = (mup / lambda) * (b[c][jj] * (double)delta(ii, r) +
                                                     b[ii][r] * (double)delta(jj, c));
                  }
                }
                for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                  d_stress->S[p][q][mode][r][c][j] += conf[p][q] * bf[var]->phi[j];
                }
              }
            }
          }
        }

        // mesh
        var = MESH_DISPLACEMENT1;
        if (pd->v[pg->imtrx][var]) {
          for (int a = 0; a < pd->Num_Dim; a++) {
            for (int k = 0; k < ei[pg->imtrx]->dof[var + a]; k++) {
              for (int ii = 0; ii < VIM; ii++) {
                for (int jj = 0; jj < VIM; jj++) {
                  stress[ii][jj] += -((d_mu->X[a][k] - d_mus->X[a][k]) / lambda) *
                                    (delta(ii, jj) - bdotb[ii][jj]);
                }
              }
            }
          }
        }

        // velocity
        var = VELOCITY1;
        if (pd->v[pg->imtrx][var]) {
          for (int a = 0; a < WIM; a++) {
            for (int k = 0; k < ei[pg->imtrx]->dof[var + a]; k++) {
              for (int ii = 0; ii < VIM; ii++) {
                for (int jj = 0; jj < VIM; jj++) {
                  stress[ii][jj] += -((d_mu->v[a][k] - d_mus->v[a][k]) / lambda) *
                                    (delta(ii, jj) - bdotb[ii][jj]);
                }
              }
            }
          }
        }

        // temperature
        var = TEMPERATURE;
        if (pd->v[pg->imtrx][var]) {
          for (int k = 0; k < ei[pg->imtrx]->dof[var]; k++) {
            for (int ii = 0; ii < VIM; ii++) {
              for (int jj = 0; jj < VIM; jj++) {
                stress[ii][jj] +=
                    -((d_mu->T[k] - d_mus->T[k]) / lambda) * (delta(ii, jj) - bdotb[ii][jj]);
              }
            }
          }
        }

        // fill
        var = FILL;
        if (pd->v[pg->imtrx][var]) {
          for (int k = 0; k < ei[pg->imtrx]->dof[var]; k++) {
            for (int ii = 0; ii < VIM; ii++) {
              for (int jj = 0; jj < VIM; jj++) {
                stress[ii][jj] +=
                    -((d_mu->F[k] - d_mus->F[k]) / lambda) * (delta(ii, jj) - bdotb[ii][jj]);
              }
            }
          }
        }
      }
    } break;
    default:
      GOMA_EH(GOMA_ERROR, "Unkown viscoelastic model %d", vn->evssModel);
    }
  }
}

void momentum_ve_stress_term(dbl mu,
                             dbl mus,
                             dbl mu_over_mu_num,
                             VISCOSITY_DEPENDENCE_STRUCT *d_mu,
                             VISCOSITY_DEPENDENCE_STRUCT *d_mus,
                             dbl d_mun_dS[MAX_MODES][DIM][DIM][MDE],
                             dbl d_mun_dG[DIM][DIM][MDE],
                             dbl stress[DIM][DIM],
                             STRESS_DEPENDENCE_STRUCT *d_stress) {
  bool compute_derivatives = (d_stress != NULL);
  dbl gamma_cont[DIM][DIM] = {{0.}};
  dbl s[DIM][DIM];
  STRESS_DEPENDENCE_STRUCT d_s_s;
  STRESS_DEPENDENCE_STRUCT *d_s = NULL;
  if (compute_derivatives) {
    d_s = &d_s_s;
    memset(d_s, 0, sizeof(STRESS_DEPENDENCE_STRUCT));
  }

  bool evss_f = is_evss_f_model(vn->evssModel);
  if (evss_f) {
    for (int a = 0; a < VIM; a++) {
      for (int b = 0; b < VIM; b++) {
        gamma_cont[a][b] = fv->G[a][b] + fv->G[b][a];
      }
    }
  }

  ve_stress_term(mu, mus, d_mu, d_mus, s, d_s);

  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
      if (evss_f) {
        stress[a][b] += -(mu - mus) * gamma_cont[a][b];
      }
      stress[a][b] += s[a][b];
    }
  }

  if (compute_derivatives) {
    int v_s[MAX_MODES][DIM][DIM];
    (void)stress_eqn_pointer(v_s);
    int var = VELOCITY1;
    if (pd->v[pg->imtrx][var]) {
      for (int a = 0; a < WIM; a++) {
        for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          for (int ii = 0; ii < VIM; ii++) {
            for (int jj = 0; jj < VIM; jj++) {
              if (evss_f) {
                d_stress->v[ii][jj][a][j] += -(d_mu->v[a][j] - d_mus->v[a][j]) * gamma_cont[ii][jj];
              }
              d_stress->v[ii][jj][a][j] += d_s->v[ii][jj][a][j];
            }
          }
        }
      }
    }

    var = MESH_DISPLACEMENT1;
    if (pd->v[pg->imtrx][var]) {
      for (int a = 0; a < pd->Num_Dim; a++) {
        for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          for (int ii = 0; ii < VIM; ii++) {
            for (int jj = 0; jj < VIM; jj++) {
              if (evss_f) {
                d_stress->X[ii][jj][a][j] += -(d_mu->X[a][j] - d_mus->X[a][j]) * gamma_cont[ii][jj];
              }
              d_stress->X[ii][jj][a][j] += d_s->X[ii][jj][a][j];
            }
          }
        }
      }
    }

    var = TEMPERATURE;
    if (pd->v[pg->imtrx][var]) {
      for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
        for (int ii = 0; ii < VIM; ii++) {
          for (int jj = 0; jj < VIM; jj++) {
            if (evss_f) {
              d_stress->T[ii][jj][j] += -(d_mu->T[j] - d_mus->T[j]) * gamma_cont[ii][jj];
            }
            d_stress->T[ii][jj][j] += d_s->T[ii][jj][j];
          }
        }
      }
    }

    var = BOND_EVOLUTION;
    if (pd->v[pg->imtrx][var]) {
      for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
        for (int ii = 0; ii < VIM; ii++) {
          for (int jj = 0; jj < VIM; jj++) {
            if (evss_f) {
              d_stress->nn[ii][jj][j] += -(d_mu->nn[j] - d_mus->nn[j]) * gamma_cont[ii][jj];
            }
            d_stress->nn[ii][jj][j] += d_s->nn[ii][jj][j];
          }
        }
      }
    }

    var = RESTIME;
    if (pd->v[pg->imtrx][var]) {
      for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
        for (int ii = 0; ii < VIM; ii++) {
          for (int jj = 0; jj < VIM; jj++) {
            if (evss_f) {
              d_stress->degrade[ii][jj][j] += -(d_mu->nn[j] - d_mus->nn[j]) * gamma_cont[ii][jj];
            }
            d_stress->degrade[ii][jj][j] += d_s->nn[ii][jj][j];
          }
        }
      }
    }

    var = FILL;
    if (pd->v[pg->imtrx][var]) {
      for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
        for (int ii = 0; ii < VIM; ii++) {
          for (int jj = 0; jj < VIM; jj++) {
            if (evss_f) {
              d_stress->F[ii][jj][j] += -(d_mu->nn[j] - d_mus->nn[j]) * gamma_cont[ii][jj];
            }
            d_stress->F[ii][jj][j] += d_s->nn[ii][jj][j];
          }
        }
      }
    }

    var = PRESSURE;
    if (pd->v[pg->imtrx][var]) {
      for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
        for (int ii = 0; ii < VIM; ii++) {
          for (int jj = 0; jj < VIM; jj++) {
            if (evss_f) {
              d_stress->P[ii][jj][j] += -(d_mu->nn[j] - d_mus->nn[j]) * gamma_cont[ii][jj];
            }
            d_stress->P[ii][jj][j] += d_s->nn[ii][jj][j];
          }
        }
      }
    }

    var = MASS_FRACTION;
    if (pd->v[pg->imtrx][var]) {
      for (int w = 0; w < pd->Num_Species_Eqn; w++) {
        for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
          for (int ii = 0; ii < VIM; ii++) {
            for (int jj = 0; jj < VIM; jj++) {
              if (evss_f) {
                d_stress->C[ii][jj][w][j] += -(d_mu->C[w][j] - d_mus->C[w][j]) * gamma_cont[ii][jj];
              }
              d_stress->C[ii][jj][w][j] += d_s->C[ii][jj][w][j];
            }
          }
        }
      }
    }

    var = POLYMER_STRESS11;
    if (pd->v[pg->imtrx][var]) {
      for (int mode = 0; mode < vn->modes; mode++) {
        for (int p = 0; p < VIM; p++) {
          for (int q = 0; q < VIM; q++) {
            var = v_s[mode][p][q];
            for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
              for (int ii = 0; ii < VIM; ii++) {
                for (int jj = 0; jj < VIM; jj++) {
                  if (evss_f) {
                    d_stress->S[ii][jj][mode][p][q][j] +=
                        mu_over_mu_num * d_mun_dS[mode][p][q][j] * (-gamma_cont[ii][jj]);
                  }
                  d_stress->S[ii][jj][mode][p][q][j] += d_s->S[ii][jj][mode][p][q][j];
                }
              }
            }
          }
        }
      }
    }

    var = VELOCITY_GRADIENT11;
    if (pd->v[pg->imtrx][var] && pd->gv[POLYMER_STRESS11]) {
      int v_g[DIM][DIM];
      v_g[0][0] = VELOCITY_GRADIENT11;
      v_g[0][1] = VELOCITY_GRADIENT12;
      v_g[1][0] = VELOCITY_GRADIENT21;
      v_g[1][1] = VELOCITY_GRADIENT22;
      v_g[0][2] = VELOCITY_GRADIENT13;
      v_g[1][2] = VELOCITY_GRADIENT23;
      v_g[2][0] = VELOCITY_GRADIENT31;
      v_g[2][1] = VELOCITY_GRADIENT32;
      v_g[2][2] = VELOCITY_GRADIENT33;
      for (int ii = 0; ii < VIM; ii++) {
        for (int jj = 0; jj < VIM; jj++) {
          for (int p = 0; p < VIM; p++) {
            for (int q = 0; q < VIM; q++) {
              var = v_g[p][q];
              for (int j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                if (evss_f) {
                  d_stress->g[ii][jj][p][q][j] +=
                      mu_over_mu_num * d_mun_dG[p][q][j] * (-gamma_cont[ii][jj]) +
                      ((mu - mus) * bf[var]->phi[j] *
                       ((double)delta(q, ii) * (double)delta(p, jj) +
                        (double)delta(p, ii) * (double)delta(q, jj)));
                }
              }
            }
          }
        }
      }
    }
  }
}

/* This stress routine does the EVSS formulation according to Fortin, 1995
 * who uses the regular stress equation and converts stress in the momentum
 * equation by adding the divergence of (g + gT).
 * Commonly denoted DEVSS-G
 */
int assemble_stress_fortin(dbl tt, /* parameter to vary time integration from
                                    * explicit (tt = 1) to implicit (tt = 0) */
                           dbl dt, /* current time step size */
                           PG_DATA *pg_data) {
  int dim, p, q, r, a, b, w, k;

  int eqn, var;
  int peqn, pvar;
  int evss_gradv = 0;

  int i, j, status, mode;
  dbl v[DIM];      /* Velocity field. */
  dbl x_dot[DIM];  /* current position field derivative wrt time. */
  dbl h3;          /* Volume element (scale factors). */
  dbl dh3dmesh_pj; /* Sensitivity to (p,j) mesh dof. */

  dbl grad_v[DIM][DIM];
  dbl gamma[DIM][DIM]; /* Shear-rate tensor based on velocity */
  dbl det_J;           /* determinant of element Jacobian */

  dbl d_det_J_dmesh_pj; /* for specific (p,j) mesh dof */

  dbl mass; /* For terms and their derivatives */
  dbl mass_a, mass_b;
  dbl advection;
  dbl advection_a, advection_b, advection_c, advection_d;
  dbl diffusion;
  dbl source;
  dbl source1;
  dbl source_a = 0, source_b = 0, source_c = 0;
  int err;
  dbl alpha = 0;      /* This is the Geisekus mobility parameter */
  dbl lambda1 = 0;    /* polymer relaxation constant */
  dbl lambda2 = 0;    /* 2nd polymer relaxation constant -- for modified Jeffreys model */
  dbl elasticMod = 0; /* elastic modulus -- needed for the modified Jeffreys model */
  dbl lambda = 0;     /* lambda1 + lambda2 -- this is just lambda1 unless using the modified
                         Jeffreys model */
  dbl d_lambda_dF[MDE];
  double xi;
  double d_xi_dF[MDE];
  dbl ucwt, lcwt; /* Upper convected derviative weight, Lower convected derivative weight */
  dbl eps = 0;    /* This is the PTT elongation parameter */
  double d_eps_dF[MDE];
  /*
   *
   * Note how carefully we avoid refering to d(phi[i])/dx[j] and refer instead
   * to the j-th component of grad_phi[j][i] so that this vector can be loaded
   * up with components that may be different in non Cartesian coordinate
   * systems.
   *
   * We will, however, insist on *orthogonal* coordinate systems, even if we
   * might permit them to be curvilinear.
   *
   * Assume all components of velocity are interpolated with the same kind
   * of basis function.
   */

  /*
   * Petrov-Galerkin weighting functions for i-th and ab-th stress residuals
   * and some of their derivatives...
   */

  dbl wt_func;

  /*
   * Interpolation functions for variables and some of their derivatives.
   */

  dbl phi_j;
  dbl wt;

  /* Variables for stress */

  int R_s[MAX_MODES][DIM][DIM];
  int v_s[MAX_MODES][DIM][DIM];
  int v_g[DIM][DIM];

  dbl s[DIM][DIM];     /* stress tensor */
  dbl s_dot[DIM][DIM]; /* stress tensor from last time step */
  dbl grad_s[DIM][DIM][DIM];
  dbl d_grad_s_dmesh[DIM][DIM][DIM][DIM]
                    [MDE]; /* derivative of grad of stress tensor for mode ve_mode */

  dbl g[DIM][DIM];      /* velocity gradient tensor */
  dbl gt[DIM][DIM];     /* transpose of velocity gradient tensor */
  dbl g_dot[DIM][DIM];  /* velocity gradient tensor time derivative */
  dbl gt_dot[DIM][DIM]; /* transpose of velocity gradient tensor time derivative */

  /* dot product tensors */

  dbl s_dot_s[DIM][DIM];
  dbl s_dot_g[DIM][DIM];
  dbl g_dot_s[DIM][DIM];
  dbl s_dot_gt[DIM][DIM];
  dbl gt_dot_s[DIM][DIM];

  dbl g_dot_g[DIM][DIM];
  dbl gt_dot_g[DIM][DIM];
  dbl gt_dot_gt[DIM][DIM];

  /* polymer viscosity and derivatives */
  dbl mup;
  VISCOSITY_DEPENDENCE_STRUCT d_mup_struct;
  VISCOSITY_DEPENDENCE_STRUCT *d_mup = &d_mup_struct;

  SARAMITO_DEPENDENCE_STRUCT d_saramito_struct;
  SARAMITO_DEPENDENCE_STRUCT *d_saramito = &d_saramito_struct;

  /* 2nd polymer viscosity -- for modified Jefreys model*/
  dbl mupJeff;

  const bool jeffreysEnabled = (vn->ConstitutiveEquation == MODIFIED_JEFFREYS);

  // todo: will want to parse necessary parameters... for now hard code
  const bool saramitoEnabled =
      (vn->ConstitutiveEquation == SARAMITO_OLDROYDB || vn->ConstitutiveEquation == SARAMITO_PTT ||
       vn->ConstitutiveEquation == SARAMITO_GIESEKUS);

  dbl saramitoCoeff = 1.;

  dbl d_mup_dv_pj;
  dbl d_mup_dmesh_pj;

  /*  shift function */
  dbl at = 0.0;
  dbl d_at_dT[MDE];
  dbl wlf_denom;

  /* constitutive equation parameters */
  dbl Z = 1.0; /* This is the factor appearing in front of the stress tensor in PTT */
  dbl dZ_dtrace = 0.0;

  /* advective terms are precalculated */
  dbl v_dot_del_s[DIM][DIM];
  dbl x_dot_del_s[DIM][DIM];

  dbl v_dot_del_g[DIM][DIM];
  dbl v_dot_del_gt[DIM][DIM];

  dbl x_dot_del_g[DIM][DIM];
  dbl x_dot_del_gt[DIM][DIM];

  dbl d_xdotdels_dm;

  dbl d_vdotdels_dm;

  dbl trace = 0.0; /* trace of the stress tensor */

  /* SUPG variables */
  dbl supg = 0;

  if (vn->evssModel == EVSS_GRADV) {
    evss_gradv = 1;
  }

  status = 0;

  eqn = R_STRESS11;

  /*
   * Bail out fast if there's nothing to do...
   */

  if (!pd->e[pg->imtrx][eqn]) {
    return (status);
  }

  /*
   * Unpack variables from structures for local convenience...
   */

  dim = pd->Num_Dim;

  wt = fv->wt;

  det_J = bf[eqn]->detJ; /* Really, ought to be mesh eqn. */

  h3 = fv->h3; /* Differential volume element (scales). */

  /* load eqn and variable number in tensor form */
  (void)stress_eqn_pointer(v_s);
  (void)stress_eqn_pointer(R_s);

  v_g[0][0] = VELOCITY_GRADIENT11;
  v_g[0][1] = VELOCITY_GRADIENT12;
  v_g[1][0] = VELOCITY_GRADIENT21;
  v_g[1][1] = VELOCITY_GRADIENT22;
  v_g[0][2] = VELOCITY_GRADIENT13;
  v_g[1][2] = VELOCITY_GRADIENT23;
  v_g[2][0] = VELOCITY_GRADIENT31;
  v_g[2][1] = VELOCITY_GRADIENT32;
  v_g[2][2] = VELOCITY_GRADIENT33;

  /*
   * Field variables...
   */
  for (a = 0; a < WIM; a++) {
    v[a] = fv->v[a];

    /* note, these are zero for steady calculations */
    x_dot[a] = 0.0;
    if (pd->TimeIntegration != STEADY && pd->gv[MESH_DISPLACEMENT1 + a]) {
      x_dot[a] = fv_dot->x[a];
    }
  }

  /*
   * In Cartesian coordinates, this velocity gradient tensor will
   * have components that are...
   *
   * 			grad_v[a][b] = d v_b
   *				       -----
   *				       d x_a
   */

  for (a = 0; a < VIM; a++) {
    for (b = 0; b < VIM; b++) {
      grad_v[a][b] = fv->grad_v[a][b];
    }
  }

  /* load up shearrate tensor based on velocity */
  for (a = 0; a < VIM; a++) {
    for (b = 0; b < VIM; b++) {
      gamma[a][b] = grad_v[a][b] + grad_v[b][a];
    }
  }

  for (a = 0; a < VIM; a++) {
    for (b = 0; b < VIM; b++) {
      if (evss_gradv) {
        g[a][b] = fv->grad_v[a][b];
        gt[a][b] = fv->grad_v[b][a];
      } else {
        g[a][b] = fv->G[a][b];
        gt[b][a] = g[a][b];
      }
    }
  }

  if (vn->wt_funcModel == GALERKIN) {
    supg = 0.;
  } else if (vn->wt_funcModel == SUPG) {
    supg = vn->wt_func;
  }

  SUPG_terms supg_terms;
  if (supg != 0.) {
    supg_tau(&supg_terms, dim, 0.0, pg_data, dt, TRUE, eqn);
  }
  /* end Petrov-Galerkin addition */

  /*  shift factor  */
  if (pd->gv[TEMPERATURE]) {
    if (vn->shiftModel == CONSTANT) {
      at = vn->shift[0];
      for (j = 0; j < ei[pg->imtrx]->dof[TEMPERATURE]; j++) {
        d_at_dT[j] = 0.;
      }
    } else if (vn->shiftModel == MODIFIED_WLF) {
      wlf_denom = vn->shift[1] + fv->T - mp->reference[TEMPERATURE];
      if (wlf_denom != 0.) {
        at = exp(vn->shift[0] * (mp->reference[TEMPERATURE] - fv->T) / wlf_denom);
        for (j = 0; j < ei[pg->imtrx]->dof[TEMPERATURE]; j++) {
          d_at_dT[j] =
              -at * vn->shift[0] * vn->shift[1] / (wlf_denom * wlf_denom) * bf[TEMPERATURE]->phi[j];
        }
      } else {
        at = 1.;
      }
      for (j = 0; j < ei[pg->imtrx]->dof[TEMPERATURE]; j++) {
        d_at_dT[j] = 0.;
      }
    }
  } else {
    at = 1.;
  }

  // if a modified Jeffreys model is being run, load time derivative of velocity gradient
  if (jeffreysEnabled) {
    (void)tensor_dot(g, g, g_dot_g, VIM);
    (void)tensor_dot(gt, gt, gt_dot_gt, VIM);
    (void)tensor_dot(gt, g, gt_dot_g, VIM);

    for (a = 0; a < VIM; a++) {
      for (b = 0; b < VIM; b++) {
        v_dot_del_g[a][b] = 0.;
        v_dot_del_gt[a][b] = 0.;
        x_dot_del_g[a][b] = 0.;
        x_dot_del_gt[a][b] = 0.;

        for (q = 0; q < dim; q++) {
          v_dot_del_g[a][b] += v[q] * fv->grad_G[q][a][b];
          v_dot_del_gt[a][b] += v[q] * fv->grad_G[q][b][a];
          x_dot_del_g[a][b] += x_dot[q] * fv->grad_G[q][a][b];
          x_dot_del_gt[a][b] += x_dot[q] * fv->grad_G[q][b][a];
        }

        if (pd->TimeIntegration != STEADY) {
          g_dot[a][b] = fv_dot->G[a][b];
        } else {
          g_dot[a][b] = 0.;
        }
        gt_dot[b][a] = g_dot[a][b];
      }
    }
  }

  /* Begin loop over modes */
  for (mode = 0; mode < vn->modes; mode++) {

    load_modal_pointers(mode, tt, dt, s, s_dot, grad_s, d_grad_s_dmesh);

    /* precalculate advective terms of form (v dot del tensor)*/

    trace = 0.0;

    for (a = 0; a < VIM; a++) {
      trace += s[a][a];
      for (b = 0; b < VIM; b++) {
        v_dot_del_s[a][b] = 0.;
        x_dot_del_s[a][b] = 0.;
        for (q = 0; q < WIM; q++) {
          v_dot_del_s[a][b] += v[q] * grad_s[q][a][b];
          x_dot_del_s[a][b] += x_dot[q] * grad_s[q][a][b];
        }
      }
    }

    /*
     * Stress tensor...(Note "anti-BSL" sign convention on deviatoric stress)
     */

    /* get polymer viscosity */
    mup = viscosity(ve[mode]->gn, gamma, d_mup);

    if (saramitoEnabled == TRUE) {
      compute_saramito_model_terms(&saramitoCoeff, d_saramito, s, ve[mode]->gn, FALSE);
    } else {
      saramitoCoeff = 1.;
      d_saramito->tau_y = 0;

      for (int i = 0; i < VIM; ++i) {
        for (int j = 0; j < VIM; ++j) {
          d_saramito->s[i][j] = 0;
        }
      }
    }

    double d_alpha_dF[MDE];
    /* get Geisekus mobility parameter */
    if (ve[mode]->alphaModel == CONSTANT) {
      alpha = ve[mode]->alpha;
    } else if (ls != NULL && ve[mode]->alphaModel == VE_LEVEL_SET) {
      double pos_alpha = ve[mode]->pos_ls.alpha;
      double neg_alpha = ve[mode]->alpha;
      double width = ls->Length_Scale;
      err = level_set_property(neg_alpha, pos_alpha, width, &alpha, d_alpha_dF);
      GOMA_EH(err, "level_set_property() failed for mobility parameter.");
    } else {
      GOMA_EH(GOMA_ERROR, "Unknown mobility parameter model");
    }

    /* get time constant */
    if (ve[mode]->time_constModel == CONSTANT) {
      lambda = ve[mode]->time_const;
    } else if (ve[mode]->time_constModel == CARREAU || ve[mode]->time_constModel == POWER_LAW) {
      lambda = mup / ve[mode]->time_const;
    } else if (ls != NULL && ve[mode]->time_constModel == VE_LEVEL_SET) {
      double pos_lambda = ve[mode]->pos_ls.time_const;
      double neg_lambda = ve[mode]->time_const;
      double width = ls->Length_Scale;
      err = level_set_property(neg_lambda, pos_lambda, width, &lambda, d_lambda_dF);
      GOMA_EH(err, "level_set_property() failed for polymer time constant.");
    }

    xi = 0;
    if (ve[mode]->xiModel == CONSTANT) {
      xi = ve[mode]->xi;
    } else if (ls != NULL && ve[mode]->xiModel == VE_LEVEL_SET) {
      double pos_xi = ve[mode]->pos_ls.xi;
      double neg_xi = ve[mode]->xi;
      double width = ls->Length_Scale;
      err = level_set_property(neg_xi, pos_xi, width, &xi, d_xi_dF);
      GOMA_EH(err, "level_set_property() failed for ptt xi parameter.");
    } else {
      GOMA_EH(GOMA_ERROR, "Unknown PTT Xi parameter model");
    }

    ucwt = 1.0 - xi / 2.0;
    lcwt = xi / 2.0;

    if (ve[mode]->epsModel == CONSTANT) {
      eps = ve[mode]->eps;
    } else if (ls != NULL && ve[mode]->epsModel == VE_LEVEL_SET) {
      double pos_eps = ve[mode]->pos_ls.eps;
      double neg_eps = ve[mode]->eps;
      double width = ls->Length_Scale;
      err = level_set_property(neg_eps, pos_eps, width, &eps, d_eps_dF);
      GOMA_EH(err, "level_set_property() failed for ptt epsilon parameter.");
    } else {
      GOMA_EH(GOMA_ERROR, "Unknown PTT Epsilon parameter model");
    }

    if (jeffreysEnabled) {
      mupJeff = ve[mode]->muJeffreys;
      // if the modified Jeffreys model is used, the parsed value of lambda is the
      // elastic modulus rather than the time consant
      elasticMod = lambda;
      lambda1 = mup / elasticMod; // mup/G
      lambda2 = mupJeff / elasticMod;
      lambda = lambda1 + lambda2;
    }

    Z = 1.0;
    dZ_dtrace = 0;
    if (vn->ConstitutiveEquation == PTT) {
      if (vn->ptt_type == PTT_LINEAR) {
        Z = 1 + eps * lambda * trace / mup;
        dZ_dtrace = eps * lambda / mup;
      } else if (vn->ptt_type == PTT_EXPONENTIAL) {
        Z = exp(eps * lambda * trace / mup);
        dZ_dtrace = Z * eps * lambda / mup;
      } else {
        GOMA_EH(GOMA_ERROR, "Unrecognized PTT Form %d", vn->ptt_type);
      }
    }

    /* get tensor dot products for future use */

    if (DOUBLE_NONZERO(alpha))
      (void)tensor_dot(s, s, s_dot_s, VIM);

    if (ucwt != 0.) {
      (void)tensor_dot(s, g, s_dot_g, VIM);
      (void)tensor_dot(gt, s, gt_dot_s, VIM);
    }

    if (lcwt != 0.) {
      (void)tensor_dot(s, gt, s_dot_gt, VIM);
      (void)tensor_dot(g, s, g_dot_s, VIM);
    }
    /*
     * Residuals_________________________________________________________________
     */

    if (af->Assemble_Residual) {
      /*
       * Assemble each component "ab" of the polymer stress equation...
       */
      for (a = 0; a < VIM; a++) {
        for (b = 0; b < VIM; b++) {

          if (a <= b) /* since the stress tensor is symmetric, only assemble the upper half */
          {
            eqn = R_s[mode][a][b];

            /*
             * In the element, there will be contributions to this many equations
             * based on the number of degrees of freedom...
             */

            for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {
              wt_func = bf[eqn]->phi[i];
              /* add Petrov-Galerkin terms as necessary */
              if (supg != 0.) {
                for (w = 0; w < dim; w++) {
                  wt_func += supg * supg_terms.supg_tau * v[w] * bf[eqn]->grad_phi[i][w];
                }
              }

              mass = 0.;

              if (pd->TimeIntegration != STEADY) {
                if (pd->e[pg->imtrx][eqn] & T_MASS) {
                  mass = s_dot[a][b];
                  mass *= wt_func * at * lambda * det_J * wt;
                  mass *= h3;
                  mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
                }
              }

              advection = 0.;
              if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                if (DOUBLE_NONZERO(lambda)) {

                  advection += v_dot_del_s[a][b] - x_dot_del_s[a][b];
                  if (ucwt != 0.)
                    advection -= ucwt * (gt_dot_s[a][b] + s_dot_g[a][b]);
                  if (lcwt != 0.)
                    advection += lcwt * (s_dot_gt[a][b] + g_dot_s[a][b]);

                  advection *= wt_func * at * lambda * det_J * wt * h3;
                  advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
                }
              }

              diffusion = 0.;
              if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
                diffusion *= det_J * wt * h3;
                diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
              }

              /*
               * Source term...
               */

              source = 0.;
              if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                // consider whether saramitoCoeff should multiply here
                source += saramitoCoeff * Z * s[a][b] - at * mup * (g[a][b] + gt[a][b]);

                if (DOUBLE_NONZERO(alpha)) {
                  source1 = (s_dot_s[a][b] / mup);

                  source1 *= alpha * lambda * saramitoCoeff;
                  source += source1;
                }

                if (jeffreysEnabled) {
                  source -= mup * lambda2 *
                            (g_dot[a][b] + gt_dot[a][b] + v_dot_del_g[a][b] + v_dot_del_gt[a][b] -
                             (g_dot_g[a][b] + 2 * gt_dot_g[a][b] + gt_dot_gt[a][b]));
                }

                source *= wt_func * det_J * h3 * wt;

                source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
              }

              /*
               * Add contributions to this residual (globally into Resid, and
               * locally into an accumulator)
               */

              lec->R[LEC_R_INDEX(upd->ep[pg->imtrx][eqn], i)] +=
                  mass + advection + diffusion + source;
            }
          }
        }
      }
    }

    /*
     * Jacobian terms...
     */

    if (af->Assemble_Jacobian) {
      dbl R_source, R_advection; /* Places to put the raw residual portions
                                    instead of constantly recalcing them */
      for (a = 0; a < VIM; a++) {
        for (b = 0; b < VIM; b++) {
          if (a <= b) /* since the stress tensor is symmetric, only assemble the upper half */
          {
            eqn = R_s[mode][a][b];
            peqn = upd->ep[pg->imtrx][eqn];

            R_advection = v_dot_del_s[a][b] - x_dot_del_s[a][b];
            if (ucwt != 0.)
              R_advection -= ucwt * (gt_dot_s[a][b] + s_dot_g[a][b]);
            if (lcwt != 0.)
              R_advection += lcwt * (s_dot_gt[a][b] + g_dot_s[a][b]);

            R_source = Z * s[a][b];

            if (DOUBLE_NONZERO(alpha))
              R_source += alpha * lambda * (s_dot_s[a][b] / mup);
            R_source *= saramitoCoeff;
            R_source += -at * mup * (g[a][b] + gt[a][b]);

            if (jeffreysEnabled) {
              R_source -= at * lambda2 * mup *
                          (g_dot[a][b] + gt_dot[a][b] + v_dot_del_g[a][b] - x_dot_del_g[a][b] +
                           v_dot_del_gt[a][b] - x_dot_del_gt[b][a] -
                           (g_dot_g[a][b] + 2 * gt_dot_g[a][b] + gt_dot_gt[a][b]));
            }

            for (i = 0; i < ei[pg->imtrx]->dof[eqn]; i++) {

              wt_func = bf[eqn]->phi[i];
              /* add Petrov-Galerkin terms as necessary */
              if (supg != 0.) {
                for (w = 0; w < dim; w++) {
                  wt_func += supg * supg_terms.supg_tau * v[w] * bf[eqn]->grad_phi[i][w];
                }
              }

              /*
               * Set up some preliminaries that are needed for the (a,i)
               * equation for bunches of (b,j) column variables...
               */

              /*
               * J_S_T
               */

              var = TEMPERATURE;
              if (pd->v[pg->imtrx][var]) {
                pvar = upd->vp[pg->imtrx][var];
                for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                  phi_j = bf[var]->phi[j];

                  mass = 0.;
                  dbl d_lambda_dT = 0;
                  if (jeffreysEnabled) {
                    d_lambda_dT = d_mup->T[j] / elasticMod;
                  }

                  if (pd->TimeIntegration != STEADY) {
                    if (pd->e[pg->imtrx][eqn] & T_MASS) {
                      mass = s_dot[a][b] * d_at_dT[j] * lambda;

                      if (jeffreysEnabled)
                        mass += s_dot[a][b] * at * d_lambda_dT;

                      mass *= wt_func * det_J * wt;
                      mass *= h3;
                      mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
                    }
                  }

                  advection = 0.;
                  if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                    if (DOUBLE_NONZERO(lambda)) {

                      advection += v_dot_del_s[a][b] - x_dot_del_s[a][b];
                      if (ucwt != 0.)
                        advection -= ucwt * (gt_dot_s[a][b] + s_dot_g[a][b]);
                      if (lcwt != 0.)
                        advection += lcwt * (s_dot_gt[a][b] + g_dot_s[a][b]);

                      advection *= d_at_dT[j] * lambda + (jeffreysEnabled ? d_lambda_dT * at : 0);
                      advection *=
                          pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)] * wt_func * det_J * wt * h3;
                    }
                  }

                  source = 0.;
                  source1 = 0.;
                  if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                    source = -(g[a][b] + gt[a][b]) * (at * d_mup->T[j] + mup * d_at_dT[j]);

                    if (jeffreysEnabled) {
                      source -= d_mup->T[j] * lambda2 *
                                (g_dot[a][b] + gt_dot[a][b] + v_dot_del_g[a][b] -
                                 x_dot_del_g[a][b] + v_dot_del_gt[a][b] - v_dot_del_gt[b][a] -
                                 (g_dot_g[a][b] + 2 * gt_dot_g[a][b] + gt_dot_gt[a][b]));
                    }
                    if (DOUBLE_NONZERO(alpha)) {
                      source1 -= s_dot_s[a][b] / (mup * mup) * d_mup->T[j];
                      source1 *= lambda * alpha * saramitoCoeff;
                      source += source1;
                    }
                    source *= wt_func * det_J * wt * h3;
                    source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
                  }

                  lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += mass + advection + source;
                }
              }

              /*
               * J_S_v
               */
              for (p = 0; p < WIM; p++) {
                var = VELOCITY1 + p;
                if (pd->v[pg->imtrx][var]) {
                  pvar = upd->vp[pg->imtrx][var];
                  for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                    phi_j = bf[var]->phi[j];
                    d_mup_dv_pj = d_mup->v[p][j];
                    dbl d_lambda_dv_pj = 0.;
                    if (jeffreysEnabled) {
                      d_lambda_dv_pj = d_mup_dv_pj / elasticMod;
                    }

                    mass = 0.;

                    if (pd->TimeIntegration != STEADY) {
                      if (pd->e[pg->imtrx][eqn] & T_MASS) {
                        if (supg != 0.) {
                          mass = supg * supg_terms.supg_tau * phi_j * bf[eqn]->grad_phi[i][p];

                          for (w = 0; w < dim; w++) {
                            mass += supg * supg_terms.d_supg_tau_dv[p][j] * v[w] *
                                    bf[eqn]->grad_phi[i][w];
                          }

                          mass *= s_dot[a][b];
                        }

                        mass *=
                            pd->etm[pg->imtrx][eqn][(LOG2_MASS)] * at * lambda * det_J * wt * h3;
                        if (jeffreysEnabled) {
                          mass += s_dot[a][b] * wt_func * pd->etm[pg->imtrx][eqn][(LOG2_MASS)] *
                                  at * d_lambda_dv_pj * det_J * wt * h3;
                        }
                      }
                    }

                    advection = 0.;

                    if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                      if (DOUBLE_NONZERO(lambda)) {
                        advection_a = phi_j * (grad_s[p][a][b]);

                        advection_a *= wt_func;

                        advection_b = 0.;
                        /* Petrov-Galerkin term */
                        if (supg != 0.) {

                          advection_b =
                              supg * supg_terms.supg_tau * phi_j * bf[eqn]->grad_phi[i][p];
                          for (w = 0; w < dim; w++) {
                            advection_b += supg * supg_terms.d_supg_tau_dv[p][j] * v[w] *
                                           bf[eqn]->grad_phi[i][w];
                          }

                          advection_b *= R_advection;
                        }

                        advection_c = 0.;
                        if (evss_gradv) {
                          if (pd->CoordinateSystem != CYLINDRICAL) {
                            if (ucwt != 0) {
                              for (k = 0; k < VIM; k++) {
                                advection_c -=
                                    ucwt * (bf[VELOCITY1 + a]->grad_phi_e[j][p][k][a] * s[k][b] +
                                            bf[VELOCITY1 + b]->grad_phi_e[j][p][k][b] * s[a][k]);
                              }
                            }
                            if (lcwt != 0.) {
                              for (k = 0; k < VIM; k++) {
                                advection_c +=
                                    lcwt * (bf[VELOCITY1 + b]->grad_phi_e[j][p][b][k] * s[a][k] +
                                            bf[VELOCITY1 + a]->grad_phi_e[j][p][a][k] * s[k][b]);
                              }
                            }
                          } else {
                            if (ucwt != 0) {
                              for (k = 0; k < VIM; k++) {
                                advection_c -=
                                    ucwt * (bf[VELOCITY1]->grad_phi_e[j][p][k][a] * s[k][b] +
                                            bf[VELOCITY1]->grad_phi_e[j][p][k][b] * s[a][k]);
                              }
                            }
                            if (lcwt != 0.) {
                              for (k = 0; k < VIM; k++) {
                                advection_c +=
                                    lcwt * (bf[VELOCITY1]->grad_phi_e[j][p][b][k] * s[a][k] +
                                            bf[VELOCITY1]->grad_phi_e[j][p][a][k] * s[k][b]);
                              }
                            }
                          }
                          advection_c *= wt_func;
                        }

                        advection = advection_a + advection_b + advection_c;
                        advection *= at * lambda * det_J * wt * h3;
                        advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];

                        advection += R_advection * d_lambda_dv_pj * wt_func * det_J * wt * h3 *
                                     pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
                      }
                    }

                    diffusion = 0.;
                    if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
                      diffusion *= det_J * wt * h3;
                      diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
                    }

                    source = 0.;

                    if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                      source_c = -at * d_mup_dv_pj * (g[a][b] + gt[a][b]);

                      if (jeffreysEnabled) {
                        source_c -= d_mup_dv_pj * at * lambda2 *
                                    (g_dot[a][b] + gt_dot[a][b] + v_dot_del_g[a][b] -
                                     x_dot_del_g[a][b] + v_dot_del_gt[a][b] - x_dot_del_gt[b][a] -
                                     (g_dot_g[a][b] + 2 * gt_dot_g[a][b] + gt_dot_gt[a][b]));

                        source_c -= lambda2 * mup *
                                    (phi_j * fv->grad_G[p][a][b] + phi_j * fv->grad_G[p][b][a]);
                      }
                      if (evss_gradv) {
                        if (pd->CoordinateSystem != CYLINDRICAL) {
                          source_c -= at * mup *
                                      (bf[VELOCITY1 + a]->grad_phi_e[j][p][a][b] +
                                       bf[VELOCITY1 + b]->grad_phi_e[j][p][b][a]);
                        } else {
                          source_c -= at * mup *
                                      (bf[VELOCITY1]->grad_phi_e[j][p][a][b] +
                                       bf[VELOCITY1]->grad_phi_e[j][p][b][a]);
                        }
                      }
                      source_c *= wt_func;

                      source_a = 0.;
                      if (DOUBLE_NONZERO(alpha)) {
                        source_a = -s_dot_s[a][b] / (mup * mup);
                        source_a *= wt_func * saramitoCoeff * alpha * lambda * d_mup_dv_pj;
                      }

                      source_b = 0.;
                      if (supg != 0.) {
                        source_b = supg * supg_terms.supg_tau * phi_j * bf[eqn]->grad_phi[i][p];

                        for (w = 0; w < dim; w++) {
                          source_b += supg * supg_terms.d_supg_tau_dv[p][j] * v[w] *
                                      bf[eqn]->grad_phi[i][w];
                        }

                        source_b *= R_source;
                      }

                      source = source_a + source_b + source_c;
                      source *= det_J * wt * h3;
                      source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
                    }

                    lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += mass + advection + diffusion + source;
                  }
                }
              }

              /*
               * J_S_c
               */
              var = MASS_FRACTION;
              if (pd->v[pg->imtrx][var]) {
                for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                  phi_j = bf[var]->phi[j];

                  for (w = 0; w < pd->Num_Species_Eqn; w++) {

                    source = 0.;

                    if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                      source_a = -at * d_mup->C[w][j] * (g[a][b] + gt[a][b]);

                      if (jeffreysEnabled) {
                        source_a -= at * d_mup->C[w][j] * lambda2 *
                                    (g_dot[a][b] + gt_dot[a][b] + v_dot_del_g[a][b] -
                                     x_dot_del_g[a][b] + v_dot_del_gt[a][b] - v_dot_del_gt[b][a] -
                                     (g_dot_g[a][b] + 2 * gt_dot_g[a][b] + gt_dot_gt[a][b]));
                      }

                      source_b = 0.;
                      if (DOUBLE_NONZERO(alpha)) {
                        source_b -= s_dot_s[a][b] / (mup * mup);
                        source_b *= alpha * lambda * saramitoCoeff * d_mup->C[w][j];
                      }
                      source = source_a + source_b;
                      source *= wt_func * det_J * wt * h3;
                      source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
                    }

                    if (w > 1) {
                      GOMA_EH(GOMA_ERROR, "Need more arrays for each species.");
                    }

                    lec->J[LEC_J_INDEX(peqn, MAX_PROB_VAR + w, i, j)] += source;
                  }
                }
              }

              /*
               * J_S_P
               */
              var = PRESSURE;
              if (pd->v[pg->imtrx][var]) {
                pvar = upd->vp[pg->imtrx][var];
                for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                  phi_j = bf[var]->phi[j];

                  source = 0.;
                  if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                    source_a += -at * d_mup->P[j] * (g[a][b] + gt[a][b]);

                    if (jeffreysEnabled) {
                      source_a -= at * d_mup->P[j] * lambda2 *
                                  (g_dot[a][b] + gt_dot[a][b] + v_dot_del_g[a][b] -
                                   x_dot_del_g[a][b] + v_dot_del_gt[a][b] - v_dot_del_gt[b][a] -
                                   (g_dot_g[a][b] + 2 * gt_dot_g[a][b] + gt_dot_gt[a][b]));
                    }

                    source_b = 0.;
                    if (DOUBLE_NONZERO(alpha)) {
                      source_b -= (s_dot_s[a][b] / (mup * mup));
                      source_b *= d_mup->P[j] * alpha * lambda * saramitoCoeff;
                    }
                    source = source_a + source_b;
                    source *= wt_func * det_J * wt * h3;
                    source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
                  }

                  lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += source;
                }
              }

              /*
               * J_S_d
               */
              for (p = 0; p < dim; p++) {
                var = MESH_DISPLACEMENT1 + p;
                if (pd->v[pg->imtrx][var]) {
                  pvar = upd->vp[pg->imtrx][var];
                  for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                    phi_j = bf[var]->phi[j];
                    d_det_J_dmesh_pj = bf[eqn]->d_det_J_dm[p][j];
                    dh3dmesh_pj = fv->dh3dq[p] * bf[var]->phi[j];
                    d_mup_dmesh_pj = d_mup->X[p][j];

                    mass = 0.;
                    mass_a = 0.;
                    mass_b = 0.;
                    if (pd->TimeIntegration != STEADY) {
                      if (pd->e[pg->imtrx][eqn] & T_MASS) {
                        mass_a = s_dot[a][b];
                        mass_a *= wt_func * (d_det_J_dmesh_pj * h3 + det_J * dh3dmesh_pj);

                        if (supg != 0.) {
                          for (w = 0; w < dim; w++) {
                            mass_b += supg * (supg_terms.supg_tau * v[w] *
                                                  bf[eqn]->d_grad_phi_dmesh[i][w][p][j] +
                                              supg_terms.d_supg_tau_dX[p][j] * v[w] *
                                                  bf[eqn]->grad_phi[i][w]);
                          }
                          mass_b *= s_dot[a][b] * h3 * det_J;
                        }

                        mass = mass_a + mass_b;
                        mass *= at * lambda * wt * pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
                      }
                    }

                    advection = 0.;

                    if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                      if (DOUBLE_NONZERO(lambda)) {
                        /*
                         * Four parts:
                         *    advection_a =
                         *    	Int ( ea.(v-xdot).d(Vv)/dmesh h3 |Jv| )
                         *
                         *    advection_b =
                         *  (i)	Int ( ea.(v-xdot).Vv h3 d(|Jv|)/dmesh )
                         *  (ii)  Int ( ea.(v-xdot).d(Vv)/dmesh h3 |Jv| )
                         *  (iii) Int ( ea.(v-xdot).Vv dh3/dmesh |Jv|   )
                         *
                         * For unsteady problems, we have an
                         * additional term
                         *
                         *    advection_c =
                         *    	Int ( ea.d(v-xdot)/dmesh.Vv h3 |Jv| )
                         */

                        advection_a = R_advection;

                        advection_a *= wt_func * (d_det_J_dmesh_pj * h3 + det_J * dh3dmesh_pj);

                        d_vdotdels_dm = 0.;
                        for (q = 0; q < WIM; q++) {
                          d_vdotdels_dm += (v[q] - x_dot[q]) * d_grad_s_dmesh[q][a][b][p][j];
                        }

                        advection_b = d_vdotdels_dm;
                        advection_b *= wt_func * det_J * h3;

                        advection_c = 0.;
                        if (pd->TimeIntegration != STEADY) {
                          if (pd->e[pg->imtrx][eqn] & T_MASS) {
                            d_xdotdels_dm = (1. + 2. * tt) * phi_j / dt * grad_s[p][a][b];

                            advection_c -= d_xdotdels_dm;

                            advection_c *= wt_func * h3 * det_J;
                          }
                        }

                        advection_d = 0.;
                        if (supg != 0.) {
                          for (w = 0; w < dim; w++) {
                            advection_d += supg * (supg_terms.supg_tau * v[w] *
                                                       bf[eqn]->d_grad_phi_dmesh[i][w][p][j] +
                                                   supg_terms.d_supg_tau_dX[p][j] * v[w] *
                                                       bf[eqn]->grad_phi[i][w]);
                          }

                          advection_d *= (R_advection)*det_J * h3;
                        }

                        advection = advection_a + advection_b + advection_c + advection_d;

                        advection *= wt * at * lambda * pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
                      }
                    }

                    diffusion = 0.;
                    if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
                      diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
                    }

                    /*
                     * Source term...
                     */

                    source = 0.;

                    if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                      source_a = R_source;
                      source_b = -at * (g[a][b] + gt[a][b]);

                      if (DOUBLE_NONZERO(alpha)) {
                        source_b += -s_dot_s[a][b] / (mup * mup) * alpha * lambda * saramitoCoeff;
                      }

                      source_a *= wt_func * (d_det_J_dmesh_pj * h3 + det_J * dh3dmesh_pj);

                      source_b *= wt_func * det_J * h3 * d_mup_dmesh_pj;

                      source_c = 0.;
                      if (supg != 0.) {
                        for (w = 0; w < dim; w++) {
                          source_c +=
                              supg *
                              (supg_terms.supg_tau * v[w] * bf[eqn]->d_grad_phi_dmesh[i][w][p][j] +
                               supg_terms.d_supg_tau_dX[p][j] * v[w] * bf[eqn]->grad_phi[i][w]);
                        }
                        source_c *= R_source * det_J * h3;
                      }

                      dbl source_jeff = 0;
                      if (jeffreysEnabled) {

                        dbl source_ja = 0;
                        dbl source_jb = 0;
                        dbl source_jc = 0;

                        source_ja -= mup * lambda2 * at *
                                     (g_dot[a][b] + gt_dot[a][b] + v_dot_del_g[a][b] -
                                      x_dot_del_g[a][b] + v_dot_del_gt[a][b] - v_dot_del_gt[b][a] -
                                      (g_dot_g[a][b] + 2 * gt_dot_g[a][b] + gt_dot_gt[a][b]));

                        source_ja *= wt_func * (d_det_J_dmesh_pj * h3 + det_J * dh3dmesh_pj);

                        dbl d_vdotdelg_dm = 0.;
                        dbl d_vdotdelgt_dm = 0.;

                        for (q = 0; q < dim; q++) {
                          d_vdotdels_dm += (v[q] - x_dot[q]) * d_grad_s_dmesh[q][a][b][p][j];
                          d_vdotdelg_dm += (v[q] - x_dot[q]) * fv->d_grad_G_dmesh[q][a][b][p][j];
                          d_vdotdelgt_dm += (v[q] - x_dot[q]) * fv->d_grad_G_dmesh[q][b][a][p][j];
                        }

                        source_jb += (lambda2 * at * mup * (d_vdotdelg_dm + d_vdotdelgt_dm));
                        source_jb -= d_mup_dmesh_pj * lambda2 *
                                     (g_dot[a][b] + gt_dot[a][b] + v_dot_del_g[a][b] -
                                      x_dot_del_g[a][b] + v_dot_del_gt[a][b] - v_dot_del_gt[b][a] -
                                      (g_dot_g[a][b] + 2 * gt_dot_g[a][b] + gt_dot_gt[a][b]));

                        source_jb *= wt_func * det_J * h3;

                        if (pd->TimeIntegration != STEADY) {
                          const dbl d_xdotdelg_dm =
                              (1. + 2. * tt) * phi_j / dt * fv->grad_G[p][a][b];
                          const dbl d_xdotdelgt_dm =
                              (1. + 2. * tt) * phi_j / dt * fv->grad_G[p][b][a];

                          source_jc -= lambda2 * mup * (d_xdotdelg_dm + d_xdotdelgt_dm);

                          source_jc *= wt_func * h3 * det_J;
                        }

                        if (supg != 0.) {
                          for (w = 0; w < dim; w++) {
                            // source_jd += supg * (supg_terms.supg_tau * v[w] *
                            //                          bf[eqn]->d_grad_phi_dmesh[i][w][p][j]
                            //                          +
                            //                      supg_terms.d_supg_tau_dX[p][j] * v[w] *
                            //                          bf[eqn]->grad_phi[i][w]);
                          }

                          // advection_d *=
                          //     -lambda2 * at * det_J * h3 *
                          //     (g_dot[a][b] + gt_dot[a][b] + v_dot_del_g[a][b] -
                          //     x_dot_del_g[a][b]
                          //     +
                          //      v_dot_del_gt[a][b] - v_dot_del_gt[b][a] -
                          //      (g_dot_g[a][b] + 2 * gt_dot_g[a][b] + gt_dot_gt[a][b]));
                        }
                        source_jeff = source_ja + source_jb + source_jc + source_jb;
                      }

                      source = source_a + source_b + source_c + source_jeff;

                      source *= wt * pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
                    }

                    lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += mass + advection + diffusion + source;
                  }
                }
              }

              /*
               * J_S_G
               */
              if (evss_gradv == 0) {
                for (p = 0; p < VIM; p++) {
                  for (q = 0; q < VIM; q++) {
                    var = v_g[p][q];

                    if (pd->v[pg->imtrx][var]) {
                      pvar = upd->vp[pg->imtrx][var];
                      for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                        phi_j = bf[var]->phi[j];
                        dbl dG[DIM][DIM] = {{0.0}};
                        dbl dGt[DIM][DIM] = {{0.0}};
                        dG[p][q] = phi_j;
                        dGt[q][p] = phi_j;
                        advection = 0.;
                        advection_a = 0.;
                        if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                          if (DOUBLE_NONZERO(lambda)) {

                            advection -= ucwt * (s[p][b] * (double)delta(a, q) +
                                                 s[a][p] * (double)delta(b, q));
                            advection += lcwt * (s[a][q] * (double)delta(p, b) +
                                                 s[q][b] * (double)delta(a, p));

                            advection *= phi_j * h3 * det_J;

                            advection *= wt_func * wt * at * lambda *
                                         pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
                          }
                        }

                        /*
                         * Diffusion...
                         */

                        diffusion = 0.;

                        if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
                          diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
                        }

                        /*
                         * Source term...
                         */

                        source = 0.;

                        if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                          source = -at * mup * phi_j *
                                   ((double)delta(a, p) * (double)delta(b, q) +
                                    (double)delta(b, p) * (double)delta(a, q));
                          if (jeffreysEnabled) {
                            dbl source_jeffrey = 0;
                            // g_dot
                            source_jeffrey += (1. + 2. * tt) * phi_j / dt * (double)delta(a, p) *
                                              (double)delta(b, q);
                            // gt_dot
                            source_jeffrey += (1. + 2. * tt) * phi_j / dt * (double)delta(a, q) *
                                              (double)delta(b, p);
                            //// v_dot_del_g
                            if ((a == p) && (b == q)) {
                              for (r = 0; r < dim; r++) {
                                source_jeffrey += (v[r] - x_dot[r]) * bf[var]->grad_phi[j][r];
                              }
                            }
                            //// v_dot_del_gt
                            if ((a == q) && (b == p)) {
                              for (r = 0; r < dim; r++) {
                                source_jeffrey += (v[r] - x_dot[r]) * bf[var]->grad_phi[j][r];
                              }
                            }
                            //// g_dot_g
                            dbl dG_dot_g[DIM][DIM] = {{0.}};
                            dbl g_dot_dG[DIM][DIM] = {{0.}};
                            tensor_dot(dG, g, dG_dot_g, VIM);
                            tensor_dot(g, dG, g_dot_dG, VIM);
                            source_jeffrey -= g_dot_dG[a][b] + dG_dot_g[a][b];
                            //// g_dot_gt
                            // source_jeffrey -=  (g[a][p] * (double)delta(b, q) + gt[q][b] *
                            // (double)delta(a, p)) *bf[var]->phi[j];
                            //// 2*gt_dot_g
                            dbl dGt_dot_g[DIM][DIM] = {{0.}};
                            dbl gt_dot_dG[DIM][DIM] = {{0.}};
                            tensor_dot(dGt, g, dGt_dot_g, VIM);
                            tensor_dot(gt, dG, gt_dot_dG, VIM);
                            source_jeffrey -= 2. * (dGt_dot_g[a][b] + gt_dot_dG[a][b]);
                            // source_jeffrey -= 2.*(gt[a][p] * (double)delta(b, q) + g[q][b]
                            // * (double)delta(a, p)) * bf[var]->phi[j];
                            //// gt_dot_gt
                            source_jeffrey -= g_dot_dG[b][a] + dG_dot_g[b][a];
                            // source_jeffrey -=  (gt[a][p] * (double)delta(b, q) + g[q][b] *
                            // (double)delta(a, p)) * bf[var]->phi[j];

                            source_jeffrey *= -mup * lambda2;
                            source += source_jeffrey;
                          }
                          source *=
                              det_J * h3 * wt_func * wt * pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
                        }

                        lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += advection + diffusion + source;
                      }
                    }
                  }
                }
              }

              /*
               * J_S_F
               */
              var = FILL;
              if (pd->v[pg->imtrx][var]) {
                pvar = upd->vp[pg->imtrx][var];
                for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                  phi_j = bf[var]->phi[j];

                  mass = 0.;

                  if (pd->TimeIntegration != STEADY) {
                    if (pd->e[pg->imtrx][eqn] & T_MASS) {

                      mass = s_dot[a][b];
                      mass *= d_lambda_dF[j];
                      mass *= wt_func * at * det_J * wt;
                      mass *= h3;
                      mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
                    }
                  }

                  advection = 0.;

                  if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                    if (d_lambda_dF[j] != 0.) {

                      advection += v_dot_del_s[a][b] - x_dot_del_s[a][b];
                      if (ucwt != 0.)
                        advection -= ucwt * (gt_dot_s[a][b] + s_dot_g[a][b]);
                      if (lcwt != 0.)
                        advection += lcwt * (s_dot_gt[a][b] + g_dot_s[a][b]);

                      advection *= d_lambda_dF[j];
                      advection *= wt_func * at * det_J * wt * h3;
                      advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
                    }
                  }

                  diffusion = 0.;

                  if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
                    /* add SU term in here when appropriate */

                    diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
                  }

                  source = 0.;

                  if (pd->e[pg->imtrx][eqn] & T_SOURCE) {

                    double invmup = 1 / mup;
                    // PTT
                    if (eps != 0) {
                      // product rule + exponential
                      source += Z *
                                ((lambda * trace * d_eps_dF[j] * invmup) +
                                 (d_lambda_dF[j] * trace * eps * invmup) -
                                 (lambda * trace * eps * d_mup->F[j] * invmup * invmup)) *
                                s[a][b];
                    }

                    source += -at * d_mup->F[j] * (g[a][b] + gt[a][b]);

                    // Giesekus
                    if (alpha != 0.) {
                      source += s_dot_s[a][b] *
                                (-alpha * lambda * d_mup->F[j] * invmup * invmup +
                                 d_alpha_dF[j] * lambda * invmup + alpha * d_lambda_dF[j] * invmup);
                    }

                    source *= wt_func * det_J * h3 * wt;

                    source *= pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
                  }

                  lec->J[LEC_J_INDEX(peqn, pvar, i, j)] += mass + advection + diffusion + source;
                }
              }

              /*
               * J_S_S
               */
              for (p = 0; p < VIM; p++) {
                for (q = 0; q < VIM; q++) {
                  var = v_s[mode][p][q];

                  if (pd->v[pg->imtrx][var]) {
                    pvar = upd->vp[pg->imtrx][var];
                    for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                      phi_j = bf[var]->phi[j];
                      mass = 0.;
                      if (pd->TimeIntegration != STEADY) {
                        if (pd->e[pg->imtrx][eqn] & T_MASS) {
                          mass = (1. + 2. * tt) * phi_j / dt * (double)delta(a, p) *
                                 (double)delta(b, q);
                          mass *= h3 * det_J;
                          mass *= wt_func * at * lambda * wt * pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
                        }
                      }

                      advection = 0.;

                      if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                        if (DOUBLE_NONZERO(lambda)) {
                          if ((a == p) && (b == q)) {
                            for (r = 0; r < WIM; r++) {
                              advection += (v[r] - x_dot[r]) * bf[var]->grad_phi[j][r];
                            }
                          }
                          advection -=
                              phi_j * ucwt *
                              (gt[a][p] * (double)delta(b, q) + g[q][b] * (double)delta(a, p));
                          advection +=
                              phi_j * lcwt *
                              (gt[q][b] * (double)delta(p, a) + g[a][p] * (double)delta(q, b));

                          advection *= h3 * det_J;

                          advection *= wt_func * wt * at * lambda *
                                       pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
                        }
                      }

                      /*
                       * Diffusion...
                       */

                      diffusion = 0.;

                      if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
                        diffusion *= det_J * wt * h3;
                        diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
                      }

                      /*
                       * Source term...
                       */

                      source = 0.;

                      if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                        source_a = Z * (double)delta(a, p) * (double)delta(b, q);
                        if (p == q)
                          source_a += s[a][b] * dZ_dtrace;
                        source_a *= saramitoCoeff;
                        // sensitivities for saramito model:
                        if (p <= q) {
                          source_a += d_saramito->s[p][q] * s[a][b] * Z;
                        }

                        source_b = 0.;
                        if (DOUBLE_NONZERO(alpha)) {
                          source_b =
                              alpha * lambda * saramitoCoeff *
                              (s[q][b] * (double)delta(a, p) + s[a][p] * (double)delta(b, q)) / mup;
                          if (p <= q) {
                            source_b +=
                                d_saramito->s[p][q] * alpha * lambda * (s_dot_s[a][b] / mup);
                          }
                        }

                        source = source_a + source_b;
                        source *= phi_j * det_J * h3 * wt_func * wt *
                                  pd->etm[pg->imtrx][eqn][(LOG2_SOURCE)];
                      }

                      lec->J[LEC_J_INDEX(peqn, pvar, i, j)] +=
                          mass + advection + diffusion + source;
                    }
                  }
                }
              }
            }
          }
        }
      }
    } /* End Assemble Jacobian */
  }   /* End loop over modes */

  return (status);
}