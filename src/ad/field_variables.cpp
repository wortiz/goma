#include "ad/field_variables.h"
#include "ad/beer_belly.h"
#include "ad/structs.h"

extern "C" {
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_eh.h"
#include "mm_fill_stress.h"
#include "mm_mp.h"
#include "rf_fem.h"
}

int ad_load_bf_grad(void) {
  int i, a, p, dofs = 0, status;
  struct Basis_Functions *bfv;

#ifdef DO_NOT_UNROLL
  int WIM;

  if ((pd->CoordinateSystem == CARTESIAN) || (pd->CoordinateSystem == CYLINDRICAL)) {
    WIM = pd->Num_Dim;
  } else {
    WIM = VIM;
  }
#endif

  status = 0;

  /* zero array for initialization */
  /*  v_length = DIM*DIM*DIM*MDE;
      init_vec_value(zero_array, 0., v_length); */
  if (ad_fv->basis.empty()) {
    ad_fv->basis.resize(V_LAST);
  }

  for (int v = V_FIRST; v < V_LAST; v++) {
    if (pd->gv[v]) {

      bfv = bf[v];
      dofs = ei[upd->matrix_index[v]]->dof[v];
      if (bfv->interpolation == I_N1) {
        dofs = bfv->shape_dof;
      }

      /* initialize variables */
      /* memset(&(bfv->d_phi[0][0]),0,siz); */

      /*
       * First load up components of the *raw* derivative vector "d_phi"
       */
      switch (pd->Num_Dim) {
      case 1:
        for (i = 0; i < dofs; i++) {
          ad_fv->basis[v].d_phi[i][0] =
              (ad_fv->B[0][0] * bfv->dphidxi[i][0] + ad_fv->B[0][1] * bfv->dphidxi[i][1]);
          ad_fv->basis[v].d_phi[i][1] = 0.0;
          ad_fv->basis[v].d_phi[i][2] = 0.0;
        }
        break;
      case 2:
        for (i = 0; i < dofs; i++) {
          ad_fv->basis[v].d_phi[i][0] =
              (ad_fv->B[0][0] * bfv->dphidxi[i][0] + ad_fv->B[0][1] * bfv->dphidxi[i][1]);
          ad_fv->basis[v].d_phi[i][1] =
              (ad_fv->B[1][0] * bfv->dphidxi[i][0] + ad_fv->B[1][1] * bfv->dphidxi[i][1]);
          ad_fv->basis[v].d_phi[i][2] = 0.0;
        }
        break;
      case 3:
        for (i = 0; i < dofs; i++) {
          ad_fv->basis[v].d_phi[i][0] =
              (ad_fv->B[0][0] * bfv->dphidxi[i][0] + ad_fv->B[0][1] * bfv->dphidxi[i][1] +
               ad_fv->B[0][2] * bfv->dphidxi[i][2]);
          ad_fv->basis[v].d_phi[i][1] =
              (ad_fv->B[1][0] * bfv->dphidxi[i][0] + ad_fv->B[1][1] * bfv->dphidxi[i][1] +
               ad_fv->B[1][2] * bfv->dphidxi[i][2]);
          ad_fv->basis[v].d_phi[i][2] =
              (ad_fv->B[2][0] * bfv->dphidxi[i][0] + ad_fv->B[2][1] * bfv->dphidxi[i][1] +
               ad_fv->B[2][2] * bfv->dphidxi[i][2]);
        }
        break;
      default:
        GOMA_EH(GOMA_ERROR, "Unexpected Dimension");
        break;
      }

      /*
       * Now, patch up the physical space gradient of this prototype
       * scalar function so scale factors are included.
       */

      /*	memset(&(bfv->grad_phi[0][0]),0,size1);  */

      for (i = 0; i < dofs; i++) {
        for (p = 0; p < WIM; p++) {
          ad_fv->basis[v].grad_phi[i][p] = (ad_fv->basis[v].d_phi[i][p]) / (fv->h[p]);
        }
      }

      for (i = 0; i < dofs; i++) {
        for (p = 0; p < VIM; p++) {
          for (a = 0; a < VIM; a++) {
            for (int q = 0; q < VIM; q++) {
              if (q == a)
                ad_fv->basis[v].grad_phi_e[i][a][p][a] = ad_fv->basis[v].grad_phi[i][p];
              else
                ad_fv->basis[v].grad_phi_e[i][a][p][q] = 0.0;
            }
          }
        }

        /* } */

        if (pd->CoordinateSystem != CARTESIAN) {
          GOMA_EH(GOMA_ERROR, "Only Cartesian coordinate system is supported, ad_load_bf_grad");
        }
      }
    } /* end of if v */
  } /* end of basis function loop. */

  return (status);
}

static inline ADType set_ad_or_dbl(dbl val, int eqn, int dof) {
  ADType tmp;
  if (ad_fv->total_ad_variables > 0 && af->Assemble_Jacobian == TRUE) {
    if (pd->gv[eqn]) {
      tmp = ADType(ad_fv->total_ad_variables, ad_fv->offset[eqn] + dof, val);
    } else {
      tmp = val;
    }
  } else {
    tmp = val;
  }
  return tmp;
}

extern "C" void fill_ad_field_variables() {
  if (ad_fv == NULL) {
    ad_fv = new AD_Field_Variables();
  }
  ad_fv->ielem = ei[pg->imtrx]->ielem;
  int num_ad_variables = 0;
  for (int i = V_FIRST; i < V_LAST; i++) {
    ad_fv->offset[i] = 0;
    // if (af->Assemble_Jacobian == TRUE) {
    if (pd->gv[i]) {
      ad_fv->offset[i] = num_ad_variables;
      num_ad_variables += ei[upd->matrix_index[i]]->dof[i];
    }
    // }
  }

  ad_fv->total_ad_variables = num_ad_variables;

  ad_beer_belly();
  ad_load_bf_grad();
  for (int p = 0; p < WIM; p++) {
    ad_fv->x_dot[p] = 0;
  }
  if (pd->gv[R_MESH1]) {
    for (int p = 0; p < WIM; p++) {
      for (int i = 0; i < ei[upd->matrix_index[R_MESH1 + p]]->dof[R_MESH1 + p]; i++) {
        ad_fv->d[p] += set_ad_or_dbl(*esp->d[p][i], R_MESH1 + p, i) * bf[R_MESH1 + p]->phi[i];
        if (pd->TimeIntegration != STEADY) {
          ADType udot = set_ad_or_dbl(*esp_dot->d[p][i], R_MESH1 + p, i);
          if (af->Assemble_Jacobian == TRUE) {
            udot.fastAccessDx(ad_fv->offset[R_MESH1 + p] + i) =
                (1. + 2. * tran->current_theta) / tran->delta_t;
          }
          ad_fv->x_dot[p] += udot * bf[R_MESH1 + p]->phi[i];
        } else {
          ad_fv->x_dot[p] = 0;
        }
      }
    }
  }

  if (pd->gv[VELOCITY1]) {
    for (int p = 0; p < WIM; p++) {
      ad_fv->v[p] = 0;
      ad_fv->v_dot[p] = 0;
      for (int i = 0; i < ei[upd->matrix_index[VELOCITY1 + p]]->dof[VELOCITY1 + p]; i++) {
        ad_fv->v[p] += set_ad_or_dbl(*esp->v[p][i], VELOCITY1 + p, i) * bf[VELOCITY1 + p]->phi[i];
        if (pd->TimeIntegration != STEADY) {
          ADType udot = set_ad_or_dbl(*esp_dot->v[p][i], VELOCITY1 + p, i);
          if (af->Assemble_Jacobian == TRUE) {
            udot.fastAccessDx(ad_fv->offset[VELOCITY1 + p] + i) =
                (1. + 2. * tran->current_theta) / tran->delta_t;
          }
          ad_fv->v_dot[p] += udot * bf[VELOCITY1 + p]->phi[i];
        } else {
          ad_fv->v_dot[p] = 0;
        }
      }
    }
    for (int p = 0; p < VIM; p++) {
      for (int q = 0; q < VIM; q++) {
        ad_fv->grad_v[p][q] = 0;

        for (int r = 0; r < WIM; r++) {
          for (int i = 0; i < ei[upd->matrix_index[VELOCITY1 + r]]->dof[VELOCITY1 + r]; i++) {
            ad_fv->grad_v[p][q] += set_ad_or_dbl(*esp->v[r][i], VELOCITY1 + r, i) *
                                   ad_fv->basis[VELOCITY1 + r].grad_phi_e[i][r][p][q];
          }
        }
      }
    }
  }

  if (pd->gv[SHEAR_RATE]) {
    ad_fv->SH = 0;
    for (int i = 0; i < ei[upd->matrix_index[SHEAR_RATE]]->dof[SHEAR_RATE]; i++) {
      ad_fv->SH += ADType(num_ad_variables, ad_fv->offset[SHEAR_RATE] + i, *esp->SH[i]) *
                   bf[SHEAR_RATE]->phi[i];
    }

    for (int q = 0; q < pd->Num_Dim; q++) {
      ad_fv->grad_eddy_nu[q] = 0;

      for (int i = 0; i < ei[upd->matrix_index[SHEAR_RATE]]->dof[SHEAR_RATE]; i++) {
        ad_fv->grad_eddy_nu[q] +=
            ADType(num_ad_variables, ad_fv->offset[SHEAR_RATE] + i, *esp->SH[i]) *
            ad_fv->basis[SHEAR_RATE].grad_phi[i][q];
      }
    }
  }

  if (pd->gv[SHELL_SAT_1]) {
    ad_fv->sh_sat_1 = 0;
    for (int i = 0; i < ei[upd->matrix_index[SHELL_SAT_1]]->dof[SHELL_SAT_1]; i++) {
      ad_fv->sh_sat_1 +=
          ADType(num_ad_variables, ad_fv->offset[SHELL_SAT_1] + i, *esp->sh_sat_1[i]) *
          bf[SHELL_SAT_1]->phi[i];
    }

    for (int q = 0; q < pd->Num_Dim; q++) {
      ad_fv->grad_sh_sat_1[q] = 0;

      for (int i = 0; i < ei[upd->matrix_index[SHELL_SAT_1]]->dof[SHELL_SAT_1]; i++) {
        ad_fv->grad_sh_sat_1[q] +=
            ADType(num_ad_variables, ad_fv->offset[SHELL_SAT_1] + i, *esp->sh_sat_1[i]) *
            ad_fv->basis[SHELL_SAT_1].grad_phi[i][q];
      }
    }
  }

  if (pd->gv[SHELL_SAT_2]) {
    ad_fv->sh_sat_2 = 0;
    for (int i = 0; i < ei[upd->matrix_index[SHELL_SAT_2]]->dof[SHELL_SAT_2]; i++) {
      ad_fv->sh_sat_2 +=
          ADType(num_ad_variables, ad_fv->offset[SHELL_SAT_2] + i, *esp->sh_sat_2[i]) *
          bf[SHELL_SAT_2]->phi[i];
    }

    for (int q = 0; q < pd->Num_Dim; q++) {
      ad_fv->grad_sh_sat_2[q] = 0;

      for (int i = 0; i < ei[upd->matrix_index[SHELL_SAT_2]]->dof[SHELL_SAT_2]; i++) {
        ad_fv->grad_sh_sat_2[q] +=
            ADType(num_ad_variables, ad_fv->offset[SHELL_SAT_2] + i, *esp->sh_sat_2[i]) *
            ad_fv->basis[SHELL_SAT_2].grad_phi[i][q];
      }
    }
  }

  if (pd->gv[SHELL_SAT_3]) {
    ad_fv->sh_sat_3 = 0;
    for (int i = 0; i < ei[upd->matrix_index[SHELL_SAT_3]]->dof[SHELL_SAT_3]; i++) {
      ad_fv->sh_sat_3 +=
          ADType(num_ad_variables, ad_fv->offset[SHELL_SAT_3] + i, *esp->sh_sat_3[i]) *
          bf[SHELL_SAT_3]->phi[i];
    }

    for (int q = 0; q < pd->Num_Dim; q++) {
      ad_fv->grad_sh_sat_3[q] = 0;

      for (int i = 0; i < ei[upd->matrix_index[SHELL_SAT_3]]->dof[SHELL_SAT_3]; i++) {
        ad_fv->grad_sh_sat_3[q] +=
            ADType(num_ad_variables, ad_fv->offset[SHELL_SAT_3] + i, *esp->sh_sat_3[i]) *
            ad_fv->basis[SHELL_SAT_3].grad_phi[i][q];
      }
    }
  }

  if (pd->gv[EDDY_NU]) {
    ad_fv->eddy_nu = 0;
    ad_fv->eddy_nu_dot = 0;
    for (int i = 0; i < ei[upd->matrix_index[EDDY_NU]]->dof[EDDY_NU]; i++) {
      ad_fv->eddy_nu += ADType(num_ad_variables, ad_fv->offset[EDDY_NU] + i, *esp->eddy_nu[i]) *
                        bf[EDDY_NU]->phi[i];

      if (pd->TimeIntegration != STEADY) {
        ADType ednudot = ADType(num_ad_variables, ad_fv->offset[EDDY_NU] + i, *esp_dot->eddy_nu[i]);
        ednudot.fastAccessDx(ad_fv->offset[EDDY_NU] + i) =
            (1. + 2. * tran->current_theta) / tran->delta_t;
        ad_fv->eddy_nu_dot += ednudot * bf[EDDY_NU]->phi[i];
      } else {
        ad_fv->eddy_nu_dot = 0;
      }
    }

    for (int q = 0; q < pd->Num_Dim; q++) {
      ad_fv->grad_eddy_nu[q] = 0;

      for (int i = 0; i < ei[upd->matrix_index[EDDY_NU]]->dof[EDDY_NU]; i++) {
        ad_fv->grad_eddy_nu[q] +=
            ADType(num_ad_variables, ad_fv->offset[EDDY_NU] + i, *esp->eddy_nu[i]) *
            ad_fv->basis[EDDY_NU].grad_phi[i][q];
      }
    }
  }

  if (pd->gv[TURB_K]) {
    ad_fv->turb_k = 0;
    ad_fv->turb_k_dot = 0;
    for (int i = 0; i < ei[upd->matrix_index[TURB_K]]->dof[TURB_K]; i++) {
      ad_fv->turb_k +=
          ADType(num_ad_variables, ad_fv->offset[TURB_K] + i, *esp->turb_k[i]) * bf[TURB_K]->phi[i];

      if (pd->TimeIntegration != STEADY) {
        ADType ednudot = ADType(num_ad_variables, ad_fv->offset[TURB_K] + i, *esp_dot->turb_k[i]);
        ednudot.fastAccessDx(ad_fv->offset[TURB_K] + i) =
            (1. + 2. * tran->current_theta) / tran->delta_t;
        ad_fv->turb_k_dot += ednudot * bf[TURB_K]->phi[i];
      } else {
        ad_fv->turb_k_dot = 0;
      }
    }

    for (int q = 0; q < pd->Num_Dim; q++) {
      ad_fv->grad_turb_k[q] = 0;

      for (int i = 0; i < ei[upd->matrix_index[TURB_K]]->dof[TURB_K]; i++) {
        ad_fv->grad_turb_k[q] +=
            ADType(num_ad_variables, ad_fv->offset[TURB_K] + i, *esp->turb_k[i]) *
            ad_fv->basis[TURB_K].grad_phi[i][q];
      }
    }
  }

  if (pd->gv[TURB_OMEGA]) {
    ad_fv->turb_omega = 0;
    ad_fv->turb_omega_dot = 0;
    for (int i = 0; i < ei[upd->matrix_index[TURB_OMEGA]]->dof[TURB_OMEGA]; i++) {
      ad_fv->turb_omega +=
          ADType(num_ad_variables, ad_fv->offset[TURB_OMEGA] + i, *esp->turb_omega[i]) *
          bf[TURB_OMEGA]->phi[i];

      if (pd->TimeIntegration != STEADY) {
        ADType ednudot =
            ADType(num_ad_variables, ad_fv->offset[TURB_OMEGA] + i, *esp_dot->turb_omega[i]);
        ednudot.fastAccessDx(ad_fv->offset[TURB_OMEGA] + i) =
            (1. + 2. * tran->current_theta) / tran->delta_t;
        ad_fv->turb_omega_dot += ednudot * bf[TURB_OMEGA]->phi[i];
      } else {
        ad_fv->turb_omega_dot = 0;
      }
    }

    for (int q = 0; q < pd->Num_Dim; q++) {
      ad_fv->grad_turb_omega[q] = 0;

      for (int i = 0; i < ei[upd->matrix_index[TURB_OMEGA]]->dof[TURB_OMEGA]; i++) {
        ad_fv->grad_turb_omega[q] +=
            ADType(num_ad_variables, ad_fv->offset[TURB_OMEGA] + i, *esp->turb_omega[i]) *
            ad_fv->basis[TURB_OMEGA].grad_phi[i][q];
      }
    }
  }

  if (pd->gv[PRESSURE]) {
    ad_fv->P = 0;
    for (int i = 0; i < ei[upd->matrix_index[PRESSURE]]->dof[PRESSURE]; i++) {
      ad_fv->P += set_ad_or_dbl(*esp->P[i], PRESSURE, i) * bf[PRESSURE]->phi[i];
    }
    for (int q = 0; q < pd->Num_Dim; q++) {
      ad_fv->grad_P[q] = 0;

      for (int i = 0; i < ei[upd->matrix_index[PRESSURE]]->dof[PRESSURE]; i++) {
        ad_fv->grad_P[q] +=
            set_ad_or_dbl(*esp->P[i], PRESSURE, i) * ad_fv->basis[PRESSURE].grad_phi[i][q];
      }
    }
  }

  if (pd->gv[FILL]) {
    ad_fv->F = 0;
    for (int i = 0; i < ei[upd->matrix_index[FILL]]->dof[FILL]; i++) {
      ad_fv->F += set_ad_or_dbl(*esp->F[i], FILL, i) * bf[FILL]->phi[i];
    }
    for (int q = 0; q < pd->Num_Dim; q++) {
      ad_fv->grad_F[q] = 0;

      for (int i = 0; i < ei[upd->matrix_index[FILL]]->dof[FILL]; i++) {
        ad_fv->grad_F[q] +=
            set_ad_or_dbl(*esp->F[i], FILL, i) * ad_fv->basis[FILL].grad_phi[i][q];
      }
    }
  }

  if (pd->gv[LUBP]) {
    ad_fv->lubp = 0;
    for (int i = 0; i < ei[upd->matrix_index[LUBP]]->dof[LUBP]; i++) {
      ad_fv->lubp += set_ad_or_dbl(*esp->lubp[i], LUBP, i) * bf[LUBP]->phi[i];
    }
    for (int q = 0; q < pd->Num_Dim; q++) {
      ad_fv->grad_lubp[q] = 0;

      for (int i = 0; i < ei[upd->matrix_index[LUBP]]->dof[LUBP]; i++) {
        ad_fv->grad_lubp[q] +=
            set_ad_or_dbl(*esp->lubp[i], LUBP, i) * ad_fv->basis[LUBP].grad_phi[i][q];
      }
    }
  }

  if (pd->gv[POLYMER_STRESS11]) {
    int v_s[MAX_MODES][DIM][DIM];
    stress_eqn_pointer(v_s);
    for (int mode = 0; mode < vn->modes; mode++) {
      for (int p = 0; p < VIM; p++) {
        for (int q = 0; q < VIM; q++) {
          ad_fv->S[mode][p][q] = 0;
          ad_fv->S_dot[mode][p][q] = 0;
          if (p <= q) {
            int v = v_s[mode][p][q];
            if (pd->gv[v]) {
              int dofs = ei[upd->matrix_index[v]]->dof[v];
              for (int i = 0; i < dofs; i++) {
                ad_fv->S[mode][p][q] +=
                    ADType(num_ad_variables, ad_fv->offset[v] + i, *esp->S[mode][p][q][i]) *
                    bf[v]->phi[i];
                if (pd->TimeIntegration != STEADY) {
                  ADType sdot =
                      ADType(num_ad_variables, ad_fv->offset[v] + i, *esp_dot->S[mode][p][q][i]);
                  sdot.fastAccessDx(ad_fv->offset[v] + i) =
                      (1. + 2. * tran->current_theta) / tran->delta_t;
                  ad_fv->S_dot[mode][p][q] += sdot * bf[v]->phi[i];
                } else {
                  ad_fv->S_dot[mode][p][q] = 0;
                }
              }
            }
            /* form the entire symmetric stress matrix for the momentum equation */
            ad_fv->S[mode][q][p] = ad_fv->S[mode][p][q];
            ad_fv->S_dot[mode][q][p] = ad_fv->S_dot[mode][p][q];
          }
          for (int r = 0; r < VIM; r++) {
            ad_fv->grad_S[mode][r][p][q] = 0.;
            int v = v_s[mode][p][q];
            int dofs = ei[upd->matrix_index[v]]->dof[v];

            for (int i = 0; i < dofs; i++) {
              if (p <= q) {
                ad_fv->grad_S[mode][r][p][q] +=
                    ADType(num_ad_variables, ad_fv->offset[v] + i, *esp->S[mode][p][q][i]) *
                    ad_fv->basis[v].grad_phi[i][r];
              } else {
                ad_fv->grad_S[mode][r][p][q] +=
                    ADType(num_ad_variables, ad_fv->offset[v] + i, *esp->S[mode][q][p][i]) *
                    ad_fv->basis[v].grad_phi[i][r];
              }
            }
          }
        }
      }
      for (int r = 0; r < pd->Num_Dim; r++) {
        ad_fv->div_S[mode][r] = 0.0;

        for (int q = 0; q < pd->Num_Dim; q++) {
          ad_fv->div_S[mode][r] += ad_fv->grad_S[mode][q][q][r];
        }
      }
    }
  }
  for (int p = 0; pd->gv[VELOCITY_GRADIENT11] && p < VIM; p++) {
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
    for (int q = 0; q < VIM; q++) {
      int v = v_g[p][q];
      if (pd->gv[v]) {
        ad_fv->G[p][q] = 0;
        int dofs = ei[upd->matrix_index[v]]->dof[v];
        for (int i = 0; i < dofs; i++) {
          ad_fv->G[p][q] +=
              ADType(num_ad_variables, ad_fv->offset[v] + i, *esp->G[p][q][i]) * bf[v]->phi[i];
        }
      }
    }
    for (int p = 0; p < VIM; p++) {
      for (int q = 0; q < VIM; q++) {
        int v = v_g[p][q];
        for (int r = 0; r < VIM; r++) {
          ad_fv->grad_G[r][p][q] = 0.0;
          int dofs = ei[upd->matrix_index[v]]->dof[v];
          for (int i = 0; i < dofs; i++) {
            ad_fv->grad_G[r][p][q] +=
                ADType(num_ad_variables, ad_fv->offset[v] + i, *esp->G[p][q][i]) *
                bf[v]->grad_phi[i][r];
          }
        }
      }
    }

    /*
     * div(G) - this is a vector!
     */
    for (int r = 0; r < pd->Num_Dim; r++) {
      ad_fv->div_G[r] = 0.0;
      for (int q = 0; q < pd->Num_Dim; q++) {
        ad_fv->div_G[r] += ad_fv->grad_G[q][q][r];
      }
    }
  }

  // if (ei[pg->imtrx]->ielem == 418) {
  //   printf("ad_fv->P = %.15f\n", ad_fv->P.val());
  // }

#if 0
  // check field variables
  for (int p = 0; p < VIM; p++) {
    if (fabs(ad_fv->v[p].val() - fv->v[p]) > 1e-14) {
      printf("diff in fv->v[%d] %.12f != %.12f\n", p, ad_fv->v[p].val(), fv->v[p]);
    }
    for (int q = 0; q < VIM; q++) {
      if (fabs(ad_fv->grad_v[p][q].val() - fv->grad_v[p][q]) > 1e-12) {
        printf("diff in fv->grad_v[%d][%d] %.12f != %.12f\n", p, q, ad_fv->grad_v[p][q].val(),
               fv->grad_v[p][q]);
      }
    }
  }
  if (fabs(ad_fv->eddy_nu.val() - fv->eddy_nu) > 1e-14) {
    printf("diff in fv->eddy_nu %.12f != %.12f\n", ad_fv->eddy_nu.val(), fv->eddy_nu);
  }
  if (fabs(ad_fv->eddy_nu_dot.val() - fv_dot->eddy_nu) > 1e-14) {
    printf("diff in fv->eddy_nu_dot %.12f != %.12f\n", ad_fv->eddy_nu_dot.val(), fv_dot->eddy_nu);
  }
  for (int p = 0; p < pd->Num_Dim; p++) {
    if (fabs(ad_fv->grad_eddy_nu[p].val() - fv->grad_eddy_nu[p]) > 1e-14) {
      printf("diff in fv->grad_eddy_nu[%d] %.12f != %.12f\n", p, ad_fv->grad_eddy_nu[p].val(),
             fv->grad_eddy_nu[p]);
    }
  }
#endif
}