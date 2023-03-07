#include "mm_fill_stress_sqrt_conf.h"
#include "mm_fill_stress.h"
#include "mm_as.h"
#include "mm_fill_stress_legacy.h"
#include "rf_fem.h"
#include "mm_mp.h"
#include "ac_stability.h"
#include "ac_stability_util.h"
#include "az_aztec.h"
#include "bc_colloc.h"
#include "el_elm.h"
#include "el_elm_info.h"
#include "el_geom.h"
#include "exo_struct.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "mm_bc.h"
#include "mm_eh.h"
#include "mm_fill_aux.h"
#include "mm_fill_fill.h"
#include "mm_fill_ls.h"
#include "mm_fill_stabilization.h"
#include "mm_fill_terms.h"
#include "mm_fill_util.h"
#include "mm_mp.h"
#include "mm_mp_const.h"
#include "mm_mp_structs.h"
#include "mm_post_def.h"
#include "mm_unknown_map.h"
#include "mm_viscosity.h"
#include "rf_allo.h"
#include "rf_bc.h"
#include "rf_bc_const.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_node_const.h"
#include "rf_solver.h"
#include "rf_solver_const.h"
#include "rf_vars_const.h"
#include "sl_util_structs.h"
#include "std.h"


void compute_a_dot_b(dbl b[DIM][DIM],
                     dbl G[DIM][DIM],
                     dbl a_dot_b[DIM][DIM],
                     dbl d_a_dot_b_db[DIM][DIM][DIM][DIM],
                     dbl d_a_dot_b_dG[DIM][DIM][DIM][DIM]) {

  if (VIM == 2) {

    dbl a12 = ((b[0][1] * G[0][0] - b[0][0] * G[0][1]) + (b[1][1] * G[1][0] - b[1][0] * G[1][1])) /
              (b[0][0] + b[1][1] + 1e-16);

    dbl a[DIM][DIM] = {{0., a12, 0.}, {-a12, 0., 0.}, {0., 0., 0.}};

    tensor_dot(a, b, a_dot_b, VIM);

    if (af->Assemble_Jacobian) {

      if (pd->v[pg->imtrx][R_GRADIENT11]) {
        for (int i = 0; i < VIM; i++) {
          for (int j = 0; j < VIM; j++) {
            dbl da12 =
                ((b[0][1] * delta(i, 0) * delta(j, 0) - b[0][0] * delta(i, 0) * delta(j, 1)) +
                 (b[1][1] * delta(i, 1) * delta(j, 0) - b[1][0] * delta(i, 1) * delta(j, 1))) /
                (b[0][0] + b[1][1] + 1e-16);

            dbl da[DIM][DIM] = {{0., da12, 0.}, {-da12, 0., 0.}, {0., 0., 0.}};

            tensor_dot(da, b, d_a_dot_b_dG[i][j], VIM);
          }
        }
      }

      for (int i = 0; i < VIM; i++) {
        for (int j = 0; j < VIM; j++) {

          dbl da12 =
              ((-delta(i, 0) * delta(j, 0) * G[0][1] +
                (delta(i, 0) * delta(j, 1) + delta(i, 1) * delta(j, 0)) * (G[0][0] - G[1][1]) +
                delta(i, 1) * delta(j, 1) * G[1][0]) -
               (1 - delta(i, 0) * delta(j, 1) - delta(i, 1) * delta(j, 0)) * a12) /
              (b[0][0] + b[1][1] + 1e-16);

          dbl da[DIM][DIM] = {{0., da12, 0.}, {-da12, 0., 0.}, {0., 0., 0.}};
          dbl db[DIM][DIM];
          for (int p = 0; p < VIM; p++) {
            for (int q = 0; q < VIM; q++) {
              db[p][q] = delta(i, p) * delta(j, q) | delta(i, q) * delta(j, p);
            }
          }
          dbl a_dot_db[DIM][DIM];
          dbl da_dot_b[DIM][DIM];

          tensor_dot(da, b, da_dot_b, VIM);
          tensor_dot(a, db, a_dot_db, VIM);
          for (int p = 0; p < VIM; p++) {
            for (int q = 0; q < VIM; q++) {
              d_a_dot_b_db[i][j][p][q] = da_dot_b[p][q] + a_dot_db[p][q];
            }
          }
        }
      }
    }

  } else { // VIM = 3
    dbl D = -b[0][1] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) -
            b[0][2] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) +
            (b[1][1] + b[2][2]) * (-b[1][2] * b[1][2] + (b[0][0] + b[1][1]) * (b[0][0] + b[2][2])) +
            1e-16;
    dbl invD = 1.0 / D;

    dbl a12 = invD * (-pow(b[0][1], 2) + (b[0][0] + b[2][2]) * (b[1][1] + b[2][2])) *
                  (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] - G[1][1] * b[0][1] +
                   G[2][0] * b[1][2] - G[2][1] * b[0][2]) -
              (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) *
                  (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] - G[1][2] * b[0][1] +
                   G[2][0] * b[2][2] - G[2][2] * b[0][2]) +
              (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) *
                  (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] - G[1][2] * b[1][1] +
                   G[2][1] * b[2][2] - G[2][2] * b[1][2]);

    dbl a13 =
        invD *
        ((-pow(b[0][2], 2) + (b[0][0] + b[1][1]) * (b[1][1] + b[2][2])) *
             (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] - G[1][2] * b[0][1] +
              G[2][0] * b[2][2] - G[2][2] * b[0][2]) +
         (-b[0][1] * b[0][2] - b[1][2] * (b[1][1] + b[2][2])) *
             (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] - G[1][1] * b[0][1] +
              G[2][0] * b[1][2] - G[2][1] * b[0][2]) -
         (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) *
             (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] - G[1][2] * b[1][1] +
              G[2][1] * b[2][2] - G[2][2] * b[1][2])) /
        (-b[0][1] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) -
         b[0][2] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) +
         (b[1][1] + b[2][2]) * (-pow(b[1][2], 2) + (b[0][0] + b[1][1]) * (b[0][0] + b[2][2])));

    dbl a23 = invD * (-pow(b[1][2], 2) + (b[0][0] + b[1][1]) * (b[0][0] + b[2][2])) *
                  (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] - G[1][2] * b[1][1] +
                   G[2][1] * b[2][2] - G[2][2] * b[1][2]) +
              (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) *
                  (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] - G[1][1] * b[0][1] +
                   G[2][0] * b[1][2] - G[2][1] * b[0][2]) -
              (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) *
                  (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] - G[1][2] * b[0][1] +
                   G[2][0] * b[2][2] - G[2][2] * b[0][2]);

    dbl a[DIM][DIM] = {
        {0.0, a12, a13},
        {-a12, 0.0, a23},
        {-a13, -a23, 0.0},
    };

    tensor_dot(a, b, a_dot_b, VIM);

    if (af->Assemble_Jacobian) {

      dbl d_a12_dG[DIM][DIM];
      dbl d_a13_dG[DIM][DIM];
      dbl d_a23_dG[DIM][DIM];
      d_a12_dG[0][0] = (-b[0][1] * (pow(b[0][1], 2) - (b[0][0] + b[2][2]) * (b[1][1] + b[2][2])) -
                        b[0][2] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2]))) *
                       invD;
      d_a13_dG[0][0] = (-b[0][1] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) -
                        b[0][2] * (pow(b[0][2], 2) - (b[0][0] + b[1][1]) * (b[1][1] + b[2][2]))) *
                       invD;
      d_a23_dG[0][0] = (b[0][1] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) -
                        b[0][2] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2])) *
                       invD;
      d_a12_dG[0][1] = (b[0][0] * (pow(b[0][1], 2) - (b[0][0] + b[2][2]) * (b[1][1] + b[2][2])) +
                        b[0][2] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2]))) *
                       invD;
      d_a13_dG[0][1] = (b[0][0] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) -
                        b[0][2] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2])) *
                       invD;
      d_a23_dG[0][1] = (-b[0][0] * b[0][1] * b[1][2] + b[0][0] * b[0][2] * b[1][1] +
                        b[0][2] * b[1][1] * b[2][2] - b[0][2] * pow(b[1][2], 2)) *
                       invD;
      d_a12_dG[1][0] = (-b[1][1] * (pow(b[0][1], 2) - (b[0][0] + b[2][2]) * (b[1][1] + b[2][2])) -
                        b[1][2] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2]))) *
                       invD;
      d_a13_dG[1][0] = (b[0][0] * b[1][1] * b[1][2] + b[0][0] * b[1][2] * b[2][2] -
                        b[0][1] * b[0][2] * b[1][1] - pow(b[0][2], 2) * b[1][2]) *
                       invD;
      d_a23_dG[1][0] = (b[1][1] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) -
                        b[1][2] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2])) *
                       invD;
      d_a12_dG[1][1] = (b[0][1] * (pow(b[0][1], 2) - (b[0][0] + b[2][2]) * (b[1][1] + b[2][2])) +
                        b[1][2] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2]))) *
                       invD;
      d_a13_dG[1][1] = (b[0][1] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) -
                        b[1][2] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2])) *
                       invD;
      d_a23_dG[1][1] = (-b[0][1] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) -
                        b[1][2] * (pow(b[1][2], 2) - (b[0][0] + b[1][1]) * (b[0][0] + b[2][2]))) *
                       invD;
      d_a12_dG[2][2] = (b[0][2] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) -
                        b[1][2] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2]))) *
                       invD;
      d_a13_dG[2][2] = (b[0][2] * (pow(b[0][2], 2) - (b[0][0] + b[1][1]) * (b[1][1] + b[2][2])) +
                        b[1][2] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2])) *
                       invD;
      d_a23_dG[2][2] = (b[0][2] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) +
                        b[1][2] * (pow(b[1][2], 2) - (b[0][0] + b[1][1]) * (b[0][0] + b[2][2]))) *
                       invD;

      if (pd->v[pg->imtrx][R_GRADIENT11]) {
        for (int i = 0; i < VIM; i++) {
          for (int j = 0; j < VIM; j++) {
            dbl da12 = d_a12_dG[i][j];
            dbl da13 = d_a13_dG[i][j];
            dbl da23 = d_a23_dG[i][j];

            dbl da[DIM][DIM] = {{0., da12, da13}, {-da12, 0., da23}, {-da13, -da23, 0.}};

            tensor_dot(da, b, d_a_dot_b_dG[i][j], VIM);
          }
        }
      }

      dbl d_D_db[DIM][DIM];

      d_D_db[0][0] = -pow(b[0][1], 2) - pow(b[0][2], 2) +
                     (b[1][1] + b[2][2]) * (2 * b[0][0] + b[1][1] + b[2][2]);
      d_D_db[0][1] = -2 * b[0][1] * (b[0][0] + b[1][1]) - 2 * b[0][2] * b[1][2];
      d_D_db[0][2] = -2 * b[0][1] * b[1][2] - 2 * b[0][2] * (b[0][0] + b[2][2]);
      d_D_db[1][0] = -2 * b[0][1] * (b[0][0] + b[1][1]) - 2 * b[0][2] * b[1][2];
      d_D_db[1][1] = -pow(b[0][1], 2) - pow(b[1][2], 2) +
                     (b[0][0] + b[1][1]) * (b[0][0] + b[2][2]) +
                     (b[0][0] + b[2][2]) * (b[1][1] + b[2][2]);
      d_D_db[1][2] = -2 * b[0][1] * b[0][2] - 2 * b[1][2] * (b[1][1] + b[2][2]);
      d_D_db[2][0] = -2 * b[0][1] * b[1][2] - 2 * b[0][2] * (b[0][0] + b[2][2]);
      d_D_db[2][1] = -2 * b[0][1] * b[0][2] - 2 * b[1][2] * (b[1][1] + b[2][2]);
      d_D_db[2][2] = -pow(b[0][2], 2) - pow(b[1][2], 2) +
                     (b[0][0] + b[1][1]) * (b[0][0] + b[2][2]) +
                     (b[0][0] + b[1][1]) * (b[1][1] + b[2][2]);

      dbl d_Da12_db[DIM][DIM];
      dbl d_Da13_db[DIM][DIM];
      dbl d_Da23_db[DIM][DIM];

      d_Da12_db[0][0] =
          G[0][1] * (pow(b[0][1], 2) - (b[0][0] + b[2][2]) * (b[1][1] + b[2][2])) +
          G[0][2] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) +
          b[0][2] * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] - G[1][2] * b[1][1] +
                     G[2][1] * b[2][2] - G[2][2] * b[1][2]) +
          (b[1][1] + b[2][2]) * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] -
                                 G[1][1] * b[0][1] + G[2][0] * b[1][2] - G[2][1] * b[0][2]);
      d_Da13_db[0][0] =
          G[0][1] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) +
          G[0][2] * (pow(b[0][2], 2) - (b[0][0] + b[1][1]) * (b[1][1] + b[2][2])) -
          b[0][1] * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] - G[1][2] * b[1][1] +
                     G[2][1] * b[2][2] - G[2][2] * b[1][2]) +
          (b[1][1] + b[2][2]) * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] -
                                 G[1][2] * b[0][1] + G[2][0] * b[2][2] - G[2][2] * b[0][2]);
      d_Da23_db[0][0] = -G[0][1] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) +
                        G[0][2] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) -
                        b[0][1] * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] -
                                   G[1][2] * b[0][1] + G[2][0] * b[2][2] - G[2][2] * b[0][2]) +
                        b[0][2] * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] -
                                   G[1][1] * b[0][1] + G[2][0] * b[1][2] - G[2][1] * b[0][2]) +
                        (2 * b[0][0] + b[1][1] + b[2][2]) *
                            (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] -
                             G[1][2] * b[1][1] + G[2][1] * b[2][2] - G[2][2] * b[1][2]);
      d_Da12_db[0][1] =
          -G[0][2] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) +
          G[1][2] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) -
          2 * b[0][1] *
              (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] - G[1][1] * b[0][1] +
               G[2][0] * b[1][2] - G[2][1] * b[0][2]) -
          b[0][2] * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] - G[1][2] * b[0][1] +
                     G[2][0] * b[2][2] - G[2][2] * b[0][2]) +
          b[1][2] * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] - G[1][2] * b[1][1] +
                     G[2][1] * b[2][2] - G[2][2] * b[1][2]) -
          (G[0][0] - G[1][1]) * (pow(b[0][1], 2) - (b[0][0] + b[2][2]) * (b[1][1] + b[2][2]));
      d_Da13_db[0][1] =
          G[0][2] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) +
          G[1][2] * (pow(b[0][2], 2) - (b[0][0] + b[1][1]) * (b[1][1] + b[2][2])) -
          b[0][2] * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] - G[1][1] * b[0][1] +
                     G[2][0] * b[1][2] - G[2][1] * b[0][2]) -
          (G[0][0] - G[1][1]) * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) -
          (b[0][0] + b[1][1]) * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] -
                                 G[1][2] * b[1][1] + G[2][1] * b[2][2] - G[2][2] * b[1][2]);
      d_Da23_db[0][1] =
          G[0][2] * (pow(b[1][2], 2) - (b[0][0] + b[1][1]) * (b[0][0] + b[2][2])) +
          G[1][2] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) +
          b[1][2] * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] - G[1][1] * b[0][1] +
                     G[2][0] * b[1][2] - G[2][1] * b[0][2]) +
          (G[0][0] - G[1][1]) * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) -
          (b[0][0] + b[1][1]) * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] -
                                 G[1][2] * b[0][1] + G[2][0] * b[2][2] - G[2][2] * b[0][2]);
      d_Da12_db[0][2] =
          G[0][1] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) +
          G[2][1] * (pow(b[0][1], 2) - (b[0][0] + b[2][2]) * (b[1][1] + b[2][2])) -
          b[0][1] * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] - G[1][2] * b[0][1] +
                     G[2][0] * b[2][2] - G[2][2] * b[0][2]) -
          (G[0][0] - G[2][2]) * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) +
          (b[0][0] + b[2][2]) * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] -
                                 G[1][2] * b[1][1] + G[2][1] * b[2][2] - G[2][2] * b[1][2]);
      d_Da13_db[0][2] =
          -G[0][1] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) +
          G[2][1] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) -
          b[0][1] * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] - G[1][1] * b[0][1] +
                     G[2][0] * b[1][2] - G[2][1] * b[0][2]) -
          2 * b[0][2] *
              (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] - G[1][2] * b[0][1] +
               G[2][0] * b[2][2] - G[2][2] * b[0][2]) -
          b[1][2] * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] - G[1][2] * b[1][1] +
                     G[2][1] * b[2][2] - G[2][2] * b[1][2]) -
          (G[0][0] - G[2][2]) * (pow(b[0][2], 2) - (b[0][0] + b[1][1]) * (b[1][1] + b[2][2]));
      d_Da23_db[0][2] =
          -G[0][1] * (pow(b[1][2], 2) - (b[0][0] + b[1][1]) * (b[0][0] + b[2][2])) -
          G[2][1] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) -
          b[1][2] * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] - G[1][2] * b[0][1] +
                     G[2][0] * b[2][2] - G[2][2] * b[0][2]) -
          (G[0][0] - G[2][2]) * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) +
          (b[0][0] + b[2][2]) * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] -
                                 G[1][1] * b[0][1] + G[2][0] * b[1][2] - G[2][1] * b[0][2]);
      d_Da12_db[1][0] =
          -G[0][2] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) +
          G[1][2] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) -
          2 * b[0][1] *
              (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] - G[1][1] * b[0][1] +
               G[2][0] * b[1][2] - G[2][1] * b[0][2]) -
          b[0][2] * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] - G[1][2] * b[0][1] +
                     G[2][0] * b[2][2] - G[2][2] * b[0][2]) +
          b[1][2] * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] - G[1][2] * b[1][1] +
                     G[2][1] * b[2][2] - G[2][2] * b[1][2]) -
          (G[0][0] - G[1][1]) * (pow(b[0][1], 2) - (b[0][0] + b[2][2]) * (b[1][1] + b[2][2]));
      d_Da13_db[1][0] =
          G[0][2] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) +
          G[1][2] * (pow(b[0][2], 2) - (b[0][0] + b[1][1]) * (b[1][1] + b[2][2])) -
          b[0][2] * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] - G[1][1] * b[0][1] +
                     G[2][0] * b[1][2] - G[2][1] * b[0][2]) -
          (G[0][0] - G[1][1]) * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) -
          (b[0][0] + b[1][1]) * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] -
                                 G[1][2] * b[1][1] + G[2][1] * b[2][2] - G[2][2] * b[1][2]);
      d_Da23_db[1][0] =
          G[0][2] * (pow(b[1][2], 2) - (b[0][0] + b[1][1]) * (b[0][0] + b[2][2])) +
          G[1][2] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) +
          b[1][2] * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] - G[1][1] * b[0][1] +
                     G[2][0] * b[1][2] - G[2][1] * b[0][2]) +
          (G[0][0] - G[1][1]) * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) -
          (b[0][0] + b[1][1]) * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] -
                                 G[1][2] * b[0][1] + G[2][0] * b[2][2] - G[2][2] * b[0][2]);
      d_Da12_db[1][1] =
          -G[1][0] * (pow(b[0][1], 2) - (b[0][0] + b[2][2]) * (b[1][1] + b[2][2])) -
          G[1][2] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) -
          b[1][2] * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] - G[1][2] * b[0][1] +
                     G[2][0] * b[2][2] - G[2][2] * b[0][2]) +
          (b[0][0] + b[2][2]) * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] -
                                 G[1][1] * b[0][1] + G[2][0] * b[1][2] - G[2][1] * b[0][2]);
      d_Da13_db[1][1] = -G[1][0] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) +
                        G[1][2] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) -
                        b[0][1] * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] -
                                   G[1][2] * b[1][1] + G[2][1] * b[2][2] - G[2][2] * b[1][2]) -
                        b[1][2] * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] -
                                   G[1][1] * b[0][1] + G[2][0] * b[1][2] - G[2][1] * b[0][2]) +
                        (b[0][0] + 2 * b[1][1] + b[2][2]) *
                            (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] -
                             G[1][2] * b[0][1] + G[2][0] * b[2][2] - G[2][2] * b[0][2]);
      d_Da23_db[1][1] =
          G[1][0] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) +
          G[1][2] * (pow(b[1][2], 2) - (b[0][0] + b[1][1]) * (b[0][0] + b[2][2])) -
          b[0][1] * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] - G[1][2] * b[0][1] +
                     G[2][0] * b[2][2] - G[2][2] * b[0][2]) +
          (b[0][0] + b[2][2]) * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] -
                                 G[1][2] * b[1][1] + G[2][1] * b[2][2] - G[2][2] * b[1][2]);
      d_Da12_db[1][2] =
          -G[1][0] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) -
          G[2][0] * (pow(b[0][1], 2) - (b[0][0] + b[2][2]) * (b[1][1] + b[2][2])) +
          b[0][1] * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] - G[1][2] * b[1][1] +
                     G[2][1] * b[2][2] - G[2][2] * b[1][2]) +
          (G[1][1] - G[2][2]) * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) -
          (b[1][1] + b[2][2]) * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] -
                                 G[1][2] * b[0][1] + G[2][0] * b[2][2] - G[2][2] * b[0][2]);
      d_Da13_db[1][2] =
          -G[1][0] * (pow(b[0][2], 2) - (b[0][0] + b[1][1]) * (b[1][1] + b[2][2])) -
          G[2][0] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) -
          b[0][2] * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] - G[1][2] * b[1][1] +
                     G[2][1] * b[2][2] - G[2][2] * b[1][2]) -
          (G[1][1] - G[2][2]) * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) -
          (b[1][1] + b[2][2]) * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] -
                                 G[1][1] * b[0][1] + G[2][0] * b[1][2] - G[2][1] * b[0][2]);
      d_Da23_db[1][2] =
          -G[1][0] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) +
          G[2][0] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) +
          b[0][1] * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] - G[1][1] * b[0][1] +
                     G[2][0] * b[1][2] - G[2][1] * b[0][2]) -
          b[0][2] * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] - G[1][2] * b[0][1] +
                     G[2][0] * b[2][2] - G[2][2] * b[0][2]) -
          2 * b[1][2] *
              (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] - G[1][2] * b[1][1] +
               G[2][1] * b[2][2] - G[2][2] * b[1][2]) -
          (G[1][1] - G[2][2]) * (pow(b[1][2], 2) - (b[0][0] + b[1][1]) * (b[0][0] + b[2][2]));
      d_Da12_db[2][0] =
          G[0][1] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) +
          G[2][1] * (pow(b[0][1], 2) - (b[0][0] + b[2][2]) * (b[1][1] + b[2][2])) -
          b[0][1] * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] - G[1][2] * b[0][1] +
                     G[2][0] * b[2][2] - G[2][2] * b[0][2]) -
          (G[0][0] - G[2][2]) * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) +
          (b[0][0] + b[2][2]) * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] -
                                 G[1][2] * b[1][1] + G[2][1] * b[2][2] - G[2][2] * b[1][2]);
      d_Da13_db[2][0] =
          -G[0][1] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) +
          G[2][1] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) -
          b[0][1] * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] - G[1][1] * b[0][1] +
                     G[2][0] * b[1][2] - G[2][1] * b[0][2]) -
          2 * b[0][2] *
              (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] - G[1][2] * b[0][1] +
               G[2][0] * b[2][2] - G[2][2] * b[0][2]) -
          b[1][2] * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] - G[1][2] * b[1][1] +
                     G[2][1] * b[2][2] - G[2][2] * b[1][2]) -
          (G[0][0] - G[2][2]) * (pow(b[0][2], 2) - (b[0][0] + b[1][1]) * (b[1][1] + b[2][2]));

      d_Da23_db[2][0] =
          -G[0][1] * (pow(b[1][2], 2) - (b[0][0] + b[1][1]) * (b[0][0] + b[2][2])) -
          G[2][1] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) -
          b[1][2] * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] - G[1][2] * b[0][1] +
                     G[2][0] * b[2][2] - G[2][2] * b[0][2]) -
          (G[0][0] - G[2][2]) * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) +
          (b[0][0] + b[2][2]) * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] -
                                 G[1][1] * b[0][1] + G[2][0] * b[1][2] - G[2][1] * b[0][2]);
      d_Da12_db[2][1] =
          -G[1][0] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) -
          G[2][0] * (pow(b[0][1], 2) - (b[0][0] + b[2][2]) * (b[1][1] + b[2][2])) +
          b[0][1] * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] - G[1][2] * b[1][1] +
                     G[2][1] * b[2][2] - G[2][2] * b[1][2]) +
          (G[1][1] - G[2][2]) * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) -
          (b[1][1] + b[2][2]) * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] -
                                 G[1][2] * b[0][1] + G[2][0] * b[2][2] - G[2][2] * b[0][2]);
      d_Da13_db[2][1] =
          -G[1][0] * (pow(b[0][2], 2) - (b[0][0] + b[1][1]) * (b[1][1] + b[2][2])) -
          G[2][0] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) -
          b[0][2] * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] - G[1][2] * b[1][1] +
                     G[2][1] * b[2][2] - G[2][2] * b[1][2]) -
          (G[1][1] - G[2][2]) * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) -
          (b[1][1] + b[2][2]) * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] -
                                 G[1][1] * b[0][1] + G[2][0] * b[1][2] - G[2][1] * b[0][2]);
      d_Da23_db[2][1] =
          -G[1][0] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) +
          G[2][0] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) +
          b[0][1] * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] - G[1][1] * b[0][1] +
                     G[2][0] * b[1][2] - G[2][1] * b[0][2]) -
          b[0][2] * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] - G[1][2] * b[0][1] +
                     G[2][0] * b[2][2] - G[2][2] * b[0][2]) -
          2 * b[1][2] *
              (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] - G[1][2] * b[1][1] +
               G[2][1] * b[2][2] - G[2][2] * b[1][2]) -
          (G[1][1] - G[2][2]) * (pow(b[1][2], 2) - (b[0][0] + b[1][1]) * (b[0][0] + b[2][2]));
      d_Da12_db[2][2] = -G[2][0] * (b[0][1] * b[0][2] + b[1][2] * (b[1][1] + b[2][2])) +
                        G[2][1] * (b[0][1] * b[1][2] + b[0][2] * (b[0][0] + b[2][2])) +
                        b[0][2] * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] -
                                   G[1][2] * b[1][1] + G[2][1] * b[2][2] - G[2][2] * b[1][2]) -
                        b[1][2] * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] -
                                   G[1][2] * b[0][1] + G[2][0] * b[2][2] - G[2][2] * b[0][2]) +
                        (b[0][0] + b[1][1] + 2 * b[2][2]) *
                            (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] -
                             G[1][1] * b[0][1] + G[2][0] * b[1][2] - G[2][1] * b[0][2]);
      d_Da13_db[2][2] =
          -G[2][0] * (pow(b[0][2], 2) - (b[0][0] + b[1][1]) * (b[1][1] + b[2][2])) -
          G[2][1] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) -
          b[1][2] * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] - G[1][1] * b[0][1] +
                     G[2][0] * b[1][2] - G[2][1] * b[0][2]) +
          (b[0][0] + b[1][1]) * (G[0][0] * b[0][2] - G[0][2] * b[0][0] + G[1][0] * b[1][2] -
                                 G[1][2] * b[0][1] + G[2][0] * b[2][2] - G[2][2] * b[0][2]);
      d_Da23_db[2][2] =
          -G[2][0] * (b[0][1] * (b[0][0] + b[1][1]) + b[0][2] * b[1][2]) -
          G[2][1] * (pow(b[1][2], 2) - (b[0][0] + b[1][1]) * (b[0][0] + b[2][2])) +
          b[0][2] * (G[0][0] * b[0][1] - G[0][1] * b[0][0] + G[1][0] * b[1][1] - G[1][1] * b[0][1] +
                     G[2][0] * b[1][2] - G[2][1] * b[0][2]) +
          (b[0][0] + b[1][1]) * (G[0][1] * b[0][2] - G[0][2] * b[0][1] + G[1][1] * b[1][2] -
                                 G[1][2] * b[1][1] + G[2][1] * b[2][2] - G[2][2] * b[1][2]);

      for (int i = 0; i < VIM; i++) {
        for (int j = 0; j < VIM; j++) {

          dbl da12 = -a12 * invD * d_D_db[i][j] + invD * d_Da12_db[i][j];
          dbl da13 = -a13 * invD * d_D_db[i][j] + invD * d_Da13_db[i][j];
          dbl da23 = -a23 * invD * d_D_db[i][j] + invD * d_Da23_db[i][j];

          dbl da[DIM][DIM] = {{0., da12, da13}, {-da12, 0., da23}, {-da13, -da23, 0.}};

          dbl db[DIM][DIM];
          for (int p = 0; p < VIM; p++) {
            for (int q = 0; q < VIM; q++) {
              db[p][q] = delta(i, p) * delta(j, q) | delta(i, q) * delta(j, p);
            }
          }
          dbl a_dot_db[DIM][DIM];
          dbl da_dot_b[DIM][DIM];

          tensor_dot(da, b, da_dot_b, VIM);
          tensor_dot(a, db, a_dot_db, VIM);
          for (int p = 0; p < VIM; p++) {
            for (int q = 0; q < VIM; q++) {
              d_a_dot_b_db[i][j][p][q] = da_dot_b[p][q] + a_dot_db[p][q];
            }
          }
        }
      }
    }
  }
}

int sqrt_conf_source(int mode,
                     dbl b[DIM][DIM],
                     dbl source_term[DIM][DIM],
                     dbl d_source_term_db[DIM][DIM][DIM][DIM]) {
  dbl binv[DIM][DIM];
  dbl d_binv_db[DIM][DIM][DIM][DIM];
  if (VIM == 2) {
    dbl det = b[0][0] * b[1][1] - b[0][1] * b[0][1] + 1e-16;
    binv[0][0] = b[1][1] / det;
    binv[0][1] = -b[0][1] / det;
    binv[1][0] = -b[0][1] / det;
    binv[1][1] = b[0][0] / det;

    for (int p = 0; p < VIM; p++) {
      for (int q = 0; q < VIM; q++) {
        dbl ddet = delta(p, 0) * delta(q, 0) * b[1][1] + b[0][0] * delta(p, 1) * delta(q, 1) -
                   2.0 * (delta(p, 0) * delta(q, 1) + delta(p, 1) * delta(q, 0)) * b[0][1];
        d_binv_db[0][0][p][q] = (det * delta(p, 1) * delta(q, 1) - ddet * b[1][1]) / (det * det);
        d_binv_db[0][1][p][q] = (-det * delta(p, 0) * delta(q, 1) + ddet * b[0][1]) / (det * det);
        d_binv_db[1][0][p][q] = (-det * delta(p, 0) * delta(q, 1) + ddet * b[1][0]) / (det * det);
        d_binv_db[1][1][p][q] = (det * delta(p, 0) * delta(q, 0) - ddet * b[0][0]) / (det * det);
      }
    }
  } else if (VIM == 3) {
    dbl det = b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1]) -
              b[0][1] * (b[1][0] * b[2][2] - b[2][0] * b[1][2]) +
              b[0][2] * (b[1][0] * b[2][1] - b[2][0] * b[1][1]) + 1e-16;

    binv[0][0] = (b[1][1] * b[2][2] - b[2][1] * b[1][2]) / (det);

    binv[0][1] = -(b[0][1] * b[2][2] - b[2][1] * b[0][2]) / (det);

    binv[0][2] = (b[0][1] * b[1][2] - b[1][1] * b[0][2]) / (det);

    binv[1][0] = -(b[1][0] * b[2][2] - b[2][0] * b[1][2]) / (det);

    binv[1][1] = (b[0][0] * b[2][2] - b[2][0] * b[0][2]) / (det);

    binv[1][2] = -(b[0][0] * b[1][2] - b[1][0] * b[0][2]) / (det);

    binv[2][0] = (b[1][0] * b[2][1] - b[1][1] * b[2][0]) / (det);

    binv[2][1] = -(b[0][0] * b[2][1] - b[2][0] * b[0][1]) / (det);

    binv[2][2] = (b[0][0] * b[1][1] - b[1][0] * b[0][1]) / (det);

    for (int p = 0; p < VIM; p++) {
      for (int q = 0; q < VIM; q++) {
        dbl db[DIM][DIM] = {{0.}};
        db[p][q] = 1.0;
        db[q][p] = 1.0;
        dbl ddet = db[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1]) +
                   b[0][0] * (db[1][1] * b[2][2] - db[1][2] * b[2][1] + b[1][1] * db[2][2] -
                              b[1][2] * db[2][1]) -
                   db[0][1] * (b[1][0] * b[2][2] - b[2][0] * b[1][2]) -
                   b[0][1] * (db[1][0] * b[2][2] - db[2][0] * b[1][2] + b[1][0] * db[2][2] -
                              b[2][0] * db[1][2]) +
                   db[0][2] * (b[1][0] * b[2][1] - b[2][0] * b[1][1]) +
                   b[0][2] * (db[1][0] * b[2][1] - db[2][0] * b[1][1] + b[1][0] * db[2][1] -
                              b[2][0] * db[1][1]);

        d_binv_db[0][0][p][q] =
            (db[1][1] * b[2][2] - db[2][1] * b[1][2] + b[1][1] * db[2][2] - b[2][1] * db[1][2]) /
            (det);
        d_binv_db[0][0][p][q] += (b[1][1] * b[2][2] - b[2][1] * b[1][2]) * -ddet / (det * det);

        d_binv_db[0][1][p][q] =
            -(db[0][1] * b[2][2] - db[2][1] * b[0][2] + b[0][1] * db[2][2] - b[2][1] * db[0][2]) /
            (det);
        d_binv_db[0][1][p][q] += -(b[0][1] * b[2][2] - b[2][1] * b[0][2]) * -ddet / (det * det);

        d_binv_db[0][2][p][q] =
            (db[0][1] * b[1][2] - db[1][1] * b[0][2] + b[0][1] * db[1][2] - b[1][1] * db[0][2]) /
            (det);
        d_binv_db[0][2][p][q] += (b[0][1] * b[1][2] - b[1][1] * b[0][2]) * -ddet / (det * det);

        d_binv_db[1][0][p][q] =
            -(db[1][0] * b[2][2] - db[2][0] * b[1][2] + b[1][0] * db[2][2] - b[2][0] * db[1][2]) /
            (det);
        d_binv_db[1][0][p][q] += -(b[1][0] * b[2][2] - b[2][0] * b[1][2]) * -ddet / (det * det);

        d_binv_db[1][1][p][q] =
            (db[0][0] * b[2][2] - db[2][0] * b[0][2] + b[0][0] * db[2][2] - b[2][0] * db[0][2]) /
            (det);
        d_binv_db[1][1][p][q] += (b[0][0] * b[2][2] - b[2][0] * b[0][2]) * -ddet / (det * det);

        d_binv_db[1][2][p][q] =
            -(db[0][0] * b[1][2] - db[1][0] * b[0][2] + b[0][0] * db[1][2] - b[1][0] * db[0][2]) /
            (det);
        d_binv_db[1][2][p][q] += -(b[0][0] * b[1][2] - b[1][0] * b[0][2]) * -ddet / (det * det);

        d_binv_db[2][0][p][q] =
            (db[1][0] * b[2][1] - db[1][1] * b[2][0] + b[1][0] * db[2][1] - b[1][1] * db[2][0]) /
            (det);
        d_binv_db[2][0][p][q] += (b[1][0] * b[2][1] - b[1][1] * b[2][0]) * -ddet / (det * det);

        d_binv_db[2][1][p][q] =
            -(db[0][0] * b[2][1] - db[2][0] * b[0][1] + b[0][0] * db[2][1] - b[2][0] * db[0][1]) /
            (det);
        d_binv_db[2][1][p][q] += -(b[0][0] * b[2][1] - b[2][0] * b[0][1]) * -ddet / (det * det);

        d_binv_db[2][2][p][q] =
            (db[0][0] * b[1][1] - db[1][0] * b[0][1] + b[0][0] * db[1][1] - b[1][0] * db[0][1]) /
            (det);
        d_binv_db[2][2][p][q] += (b[0][0] * b[1][1] - b[1][0] * b[0][1]) * -ddet / (det * det);
      }
    }
  } else {
    GOMA_EH(GOMA_ERROR, "Unknown VIM = %d for SQRT conformation tensor", VIM);
  }

  switch (vn->ConstitutiveEquation) {
  case OLDROYDB: {
    for (int ii = 0; ii < VIM; ii++) {
      for (int jj = 0; jj < VIM; jj++) {
        source_term[ii][jj] = -0.5 * (binv[ii][jj] - b[ii][jj]);
      }
    }
    if (af->Assemble_Jacobian) {
      for (int ii = 0; ii < VIM; ii++) {
        for (int jj = 0; jj < VIM; jj++) {
          for (int p = 0; p < VIM; p++) {
            for (int q = 0; q < VIM; q++) {
              d_source_term_db[ii][jj][p][q] =
                  -0.5 * (d_binv_db[ii][jj][p][q] - delta(ii, p) * delta(jj, q));
            }
          }
        }
      }
    }
  } break;
  case PTT: {

    dbl d_trace_db[DIM][DIM] = {{0.0}};

    dbl trace = 0;
    for (int i = 0; i < VIM; i++) {
      for (int j = 0; j < VIM; j++) {
        trace += b[i][j] * b[i][j];
      }
    }

    if (af->Assemble_Jacobian) {
      for (int p = 0; p < VIM; p++) {
        for (int q = 0; q < VIM; q++) {
          d_trace_db[p][q] = 0.0;
          for (int i = 0; i < VIM; i++) {
            for (int j = 0; j < VIM; j++) {
              d_trace_db[p][q] +=
                  2.0 * b[i][j] * (delta(p, i) * delta(q, j) | delta(p, j) * delta(q, i));
            }
          }
        }
      }
    }

    dbl Z = 1.0;
    dbl dZ_dtrace = 0;

    // PTT exponent
    eps = ve[mode]->eps;

    if (vn->ptt_type == PTT_LINEAR) {
      Z = 1 + eps * (trace - (double)VIM);
      dZ_dtrace = eps;
    } else if (vn->ptt_type == PTT_EXPONENTIAL) {
      Z = exp(eps * (trace - (double)VIM));
      dZ_dtrace = eps * Z;
    } else {
      GOMA_EH(GOMA_ERROR, "Unrecognized PTT Form %d", vn->ptt_type);
    }

    for (int ii = 0; ii < VIM; ii++) {
      for (int jj = 0; jj < VIM; jj++) {
        source_term[ii][jj] = -0.5 * Z * (binv[ii][jj] - b[ii][jj]);
      }
    }
    if (af->Assemble_Jacobian) {
      for (int ii = 0; ii < VIM; ii++) {
        for (int jj = 0; jj < VIM; jj++) {
          for (int p = 0; p < VIM; p++) {
            for (int q = 0; q < VIM; q++) {
              d_source_term_db[ii][jj][p][q] =
                  -0.5 * Z * (d_binv_db[ii][jj][p][q] - delta(ii, p) * delta(jj, q)) -
                  0.5 * dZ_dtrace * d_trace_db[p][q] * (binv[ii][jj] - b[ii][jj]);
            }
          }
        }
      }
    }
  } break;
  default:
    GOMA_EH(GOMA_ERROR, "Unknown Constitutive equation form for SQRT_CONF");
    break;
  }

  return GOMA_SUCCESS;
}

int assemble_stress_sqrt_conf(dbl tt, /* parameter to vary time integration from
                                       * explicit (tt = 1) to implicit (tt = 0) */
                              dbl dt, /* current time step size */
                              PG_DATA *pg_data) {
  int dim, p, q, r, w;

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
  dbl source_a = 0, source_b = 0, source_c = 0;
  int err;
  dbl alpha = 0;  /* This is the Geisekus mobility parameter */
  dbl lambda = 0; /* polymer relaxation constant */
  dbl d_lambda_dF[MDE];
  double xi;
  double d_xi_dF[MDE];
  dbl eps = 0; /* This is the PTT elongation parameter */
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

  dbl b[DIM][DIM];     /* stress tensor */
  dbl b_dot[DIM][DIM]; /* stress tensor from last time step */
  dbl grad_b[DIM][DIM][DIM];
  dbl d_grad_b_dmesh[DIM][DIM][DIM][DIM]
                    [MDE]; /* derivative of grad of stress tensor for mode ve_mode */

  dbl g[DIM][DIM];  /* velocity gradient tensor */
  dbl gt[DIM][DIM]; /* transpose of velocity gradient tensor */

  /* dot product tensors */

  dbl s_dot_s[DIM][DIM];
  dbl b_dot_g[DIM][DIM];

  /* polymer viscosity and derivatives */
  dbl mup;
  VISCOSITY_DEPENDENCE_STRUCT d_mup_struct;
  VISCOSITY_DEPENDENCE_STRUCT *d_mup = &d_mup_struct;

  const bool saramitoEnabled =
      (vn->ConstitutiveEquation == SARAMITO_OLDROYDB || vn->ConstitutiveEquation == SARAMITO_PTT ||
       vn->ConstitutiveEquation == SARAMITO_GIESEKUS);

  dbl saramitoCoeff = 1.;

  if (saramitoEnabled) {
    GOMA_EH(GOMA_ERROR, "Saramito not available for SQRT_CONF");
  }

  /*  shift function */
  dbl at = 0.0;
  dbl d_at_dT[MDE];
  dbl wlf_denom;

  /* advective terms are precalculated */
  dbl v_dot_del_b[DIM][DIM];
  dbl x_dot_del_b[DIM][DIM];

  dbl d_xdotdels_dm;

  dbl d_vdotdels_dm;

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
  for (int a = 0; a < WIM; a++) {
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

  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
      grad_v[a][b] = fv->grad_v[a][b];
    }
  }

  /* load up shearrate tensor based on velocity */
  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
      gamma[a][b] = grad_v[a][b] + grad_v[b][a];
    }
  }

  for (int a = 0; a < VIM; a++) {
    for (int b = 0; b < VIM; b++) {
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
    supg_tau(&supg_terms, dim, 1e-14, pg_data, dt, TRUE, eqn);
  }
  /* end Petrov-Galerkin addition */
  dbl yzbeta_factor = 0.0;
  if (shock_is_yzbeta(vn->shockcaptureModel)) {
    yzbeta_factor = vn->shockcapture;
  } else if (vn->shockcaptureModel != SC_NONE) {
    GOMA_EH(GOMA_ERROR, "Unknown shock capture model, only YZBETA supported for SQRT_CONF");
  }

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

  /* Begin loop over modes */
  for (mode = 0; mode < vn->modes; mode++) {

    load_modal_pointers(mode, tt, dt, b, b_dot, grad_b, d_grad_b_dmesh);

    /* precalculate advective terms of form (v dot del tensor)*/

    /*
     * Stress tensor...(Note "anti-BSL" sign convention on deviatoric stress)
     */
    for (int ii = 0; ii < VIM; ii++) {
      for (int jj = 0; jj < VIM; jj++) {
        v_dot_del_b[ii][jj] = 0.;
        x_dot_del_b[ii][jj] = 0.;
        for (q = 0; q < WIM; q++) {
          v_dot_del_b[ii][jj] += v[q] * grad_b[q][ii][jj];
          x_dot_del_b[ii][jj] += x_dot[q] * grad_b[q][ii][jj];
        }
      }
    }

    /* get polymer viscosity */
    mup = viscosity(ve[mode]->gn, gamma, d_mup);

    if (saramitoEnabled == TRUE) {
      GOMA_EH(GOMA_ERROR, "Saramito not enabled sqrt");
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

    if (DOUBLE_NONZERO(xi)) {
      GOMA_EH(GOMA_ERROR, "PTT Xi parameter currently required to be 0 for SQRT_CONF");
    }

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

    (void)tensor_dot(b, g, b_dot_g, VIM);

    dbl a_dot_b[DIM][DIM];
    dbl d_a_dot_b_db[DIM][DIM][DIM][DIM];
    dbl d_a_dot_b_dG[DIM][DIM][DIM][DIM];

    compute_a_dot_b(b, g, a_dot_b, d_a_dot_b_db, d_a_dot_b_dG);

    dbl source_term[DIM][DIM];
    dbl d_source_term_db[DIM][DIM][DIM][DIM];
    sqrt_conf_source(mode, b, source_term, d_source_term_db);

    // YZBeta Shock capturing terms
    dbl Y_inv = 1.0;
    dbl mag_b = 0;
    dbl kdc[DIM][DIM];
    dbl he = 0.0;
    dbl Z[DIM][DIM];
    dbl scaling_grad_s[DIM][DIM];
    if (shock_is_yzbeta(vn->shockcaptureModel)) {
      for (int m = 0; m < VIM; m++) {
        for (int n = 0; n < VIM; n++) {
          Z[m][n] = 0;
          Z[m][n] = b_dot[m][n];
          Z[m][n] += 1e-16 + v_dot_del_b[m][n] - x_dot_del_b[m][n];
          Z[m][n] -= b_dot_g[m][n];
          Z[m][n] -= a_dot_b[m][n];
          Z[m][n] *= at * lambda;
          Z[m][n] += source_term[m][n];
          scaling_grad_s[m][n] = 1e-16;
          for (int i = 0; i < dim; i++) {
            scaling_grad_s[m][n] += grad_b[i][m][n] * grad_b[i][m][n];
          }
        }
      }

      dbl tmp = 0;
      for (int q = 0; q < VIM; q++) {
        for (int m = 0; m < VIM; m++) {
          tmp += b[q][m] * b[q][m];
        }
      }
      mag_b = sqrt(tmp) + 1e-16;

      for (int q = 0; q < ei[pg->imtrx]->dof[eqn]; q++) {
        dbl tmp = 0;
        for (int w = 0; w < dim; w++) {
          tmp += bf[eqn]->grad_phi[q][w] * bf[eqn]->grad_phi[q][w];
        }
        he += 1.0 / sqrt(tmp);
      }
      for (int m = 0; m < VIM; m++) {
        for (int n = 0; n < VIM; n++) {
          kdc[m][n] = 0;
          if (vn->shockcaptureModel == SC_YZBETA || vn->shockcaptureModel == YZBETA_ONE ||
              vn->shockcaptureModel == YZBETA_MIXED) {
            kdc[m][n] += fabs(Z[m][n]) * Y_inv * he * he / mag_b;
          }
          if (vn->shockcaptureModel == SC_YZBETA || vn->shockcaptureModel == YZBETA_TWO ||
              vn->shockcaptureModel == YZBETA_MIXED) {
            kdc[m][n] += fabs(Z[m][n]) * Y_inv * (1.0 / sqrt(scaling_grad_s[m][n])) * he / mag_b;
          }
          if (vn->shockcaptureModel == SC_YZBETA || vn->shockcaptureModel == YZBETA_MIXED) {
            kdc[m][n] *= 0.5;
          }
        }
      }
    }
    /*
     * Residuals_________________________________________________________________
     */

    if (af->Assemble_Residual) {
      /*
       * Assemble each component "ab" of the polymer stress equation...
       */
      for (int ii = 0; ii < VIM; ii++) {
        for (int jj = 0; jj < VIM; jj++) {

          if (ii <= jj) /* since the stress tensor is symmetric, only assemble the upper half */
          {
            eqn = R_s[mode][ii][jj];

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
                  mass = b_dot[ii][jj];
                  mass *= wt_func * at * lambda * det_J * wt;
                  mass *= h3;
                  mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
                }
              }

              advection = 0.;
              if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                if (DOUBLE_NONZERO(lambda)) {

                  advection += v_dot_del_b[ii][jj] - x_dot_del_b[ii][jj];
                  advection -= b_dot_g[ii][jj];
                  advection -= a_dot_b[ii][jj];
                  advection *= wt_func * at * lambda * det_J * wt * h3;
                  advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
                }
              }

              diffusion = 0.;
              if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
                if (shock_is_yzbeta(vn->shockcaptureModel)) {
                  dbl scale = kdc[ii][jj];
                  // if supg choose max kdc/supg_tau
                  if (supg > 0) {
                    dbl tmp = kdc[ii][jj];
                    dbl tau = supg_terms.supg_tau;
                    scale = 1.0 / (sqrt(1.0 / (tmp * tmp) + 1.0 / (tau * tau)) + 1e-16);
                  }
                  for (int r = 0; r < VIM; r++) {
                    diffusion += scale * grad_b[r][ii][jj] * bf[eqn]->grad_phi[i][r];
                  }
                  diffusion *= yzbeta_factor * det_J * wt * h3;
                  diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
                }
              }

              /*
               * Source term...
               */

              source = 0.;
              if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                // consider whether saramitoCoeff should multiply here
                source = source_term[ii][jj];
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
      for (int ii = 0; ii < VIM; ii++) {
        for (int jj = 0; jj < VIM; jj++) {
          if (ii <= jj) /* since the stress tensor is symmetric, only assemble the upper half */
          {
            eqn = R_s[mode][ii][jj];
            peqn = upd->ep[pg->imtrx][eqn];

            R_advection = v_dot_del_b[ii][jj] - x_dot_del_b[ii][jj];
            R_advection -= b_dot_g[ii][jj];
            R_advection -= a_dot_b[ii][jj];

            R_source = source_term[ii][jj];

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

                  if (pd->TimeIntegration != STEADY) {
                    if (pd->e[pg->imtrx][eqn] & T_MASS) {
                      mass = b_dot[ii][jj];
                      mass *= wt_func * d_at_dT[j] * lambda * det_J * wt;
                      mass *= h3;
                      mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
                    }
                  }

                  advection = 0.;
                  if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                    if (DOUBLE_NONZERO(lambda)) {

                      advection += R_advection;

                      advection *= wt_func * d_at_dT[j] * lambda * det_J * wt * h3;
                      advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
                    }
                  }

                  source = 0.;
                  if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                    source = 0;
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

                    mass = 0.;

                    if (pd->TimeIntegration != STEADY) {
                      if (pd->e[pg->imtrx][eqn] & T_MASS) {
                        if (supg != 0.) {
                          mass = supg * supg_terms.supg_tau * phi_j * bf[eqn]->grad_phi[i][p];

                          for (w = 0; w < dim; w++) {
                            mass += supg * supg_terms.d_supg_tau_dv[p][j] * v[w] *
                                    bf[eqn]->grad_phi[i][w];
                          }

                          mass *= b_dot[ii][jj];
                        }

                        mass *=
                            pd->etm[pg->imtrx][eqn][(LOG2_MASS)] * at * lambda * det_J * wt * h3;
                      }
                    }

                    advection = 0.;

                    if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                      if (DOUBLE_NONZERO(lambda)) {
                        advection_a = phi_j * (grad_b[p][ii][jj]);

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
                        advection = advection_a + advection_b + advection_c;
                        advection *= at * lambda * det_J * wt * h3;
                        advection *= pd->etm[pg->imtrx][eqn][(LOG2_ADVECTION)];
                      }
                    }

                    diffusion = 0.;
                    if (pd->e[pg->imtrx][eqn] & T_DIFFUSION) {
                      if (shock_is_yzbeta(vn->shockcaptureModel)) {
                        dbl dZi = phi_j * (grad_b[p][ii][jj]);
                        dZi *= at * lambda;
                        dbl sign = SGN(Z[ii][jj]);
                        dbl dZ = dZi * sign;
                        dbl dkdc = 0;
                        if (vn->shockcaptureModel == SC_YZBETA ||
                            vn->shockcaptureModel == YZBETA_ONE ||
                            vn->shockcaptureModel == YZBETA_MIXED) {
                          dkdc += Y_inv * dZ * he * he / mag_b;
                        }
                        if (vn->shockcaptureModel == SC_YZBETA ||
                            vn->shockcaptureModel == YZBETA_TWO ||
                            vn->shockcaptureModel == YZBETA_MIXED) {
                          dkdc += Y_inv * dZ * (1.0 / sqrt(scaling_grad_s[ii][jj])) * he / mag_b;
                        }
                        if (vn->shockcaptureModel == SC_YZBETA ||
                            vn->shockcaptureModel == YZBETA_MIXED) {
                          dkdc *= 0.5;
                        }
                        dbl dscale = dkdc;
                        if (supg > 0) {
                          dbl tmp = kdc[ii][jj];
                          dbl tau = supg_terms.supg_tau;
                          dbl dtau = supg_terms.d_supg_tau_dv[p][j];
                          dbl scale = 1.0 / (sqrt(1.0 / (tmp * tmp) + 1.0 / (tau * tau)) + 1e-16);
                          dscale = scale * (dkdc / pow(kdc[ii][jj], 3) + dtau / pow(dtau, 3)) *
                                   scale * scale;
                        }
                        for (int r = 0; r < VIM; r++) {
                          diffusion += dscale * grad_b[r][ii][jj] * bf[eqn]->grad_phi[i][r];
                        }
                        diffusion *= yzbeta_factor * det_J * wt * h3;
                        diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
                      }
                    }

                    source = 0.;

                    if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                      source_c = 0;

                      source_a = 0.;

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
                      source_a = -at * d_mup->C[w][j] * (g[ii][jj] + gt[ii][jj]);

                      source_b = 0.;
                      if (DOUBLE_NONZERO(alpha)) {
                        source_b -= s_dot_s[ii][jj] / (mup * mup);
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

                    mass = 0.;
                    mass_a = 0.;
                    mass_b = 0.;
                    if (pd->TimeIntegration != STEADY) {
                      if (pd->e[pg->imtrx][eqn] & T_MASS) {
                        mass_a = b_dot[ii][jj];
                        mass_a *= wt_func * (d_det_J_dmesh_pj * h3 + det_J * dh3dmesh_pj);

                        if (supg != 0.) {
                          for (w = 0; w < dim; w++) {
                            mass_b += supg * (supg_terms.supg_tau * v[w] *
                                                  bf[eqn]->d_grad_phi_dmesh[i][w][p][j] +
                                              supg_terms.d_supg_tau_dX[p][j] * v[w] *
                                                  bf[eqn]->grad_phi[i][w]);
                          }
                          mass_b *= b_dot[ii][jj] * h3 * det_J;
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
                          d_vdotdels_dm += (v[q] - x_dot[q]) * d_grad_b_dmesh[q][ii][jj][p][j];
                        }

                        advection_b = d_vdotdels_dm;
                        advection_b *= wt_func * det_J * h3;

                        advection_c = 0.;
                        if (pd->TimeIntegration != STEADY) {
                          if (pd->e[pg->imtrx][eqn] & T_MASS) {
                            d_xdotdels_dm = (1. + 2. * tt) * phi_j / dt * grad_b[p][ii][jj];

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
                      if (shock_is_yzbeta(vn->shockcaptureModel)) {
                        dbl dZi = 0;
                        for (int q = 0; q < WIM; q++) {
                          dZi += (v[q] - x_dot[q]) * d_grad_b_dmesh[q][ii][jj][p][j];
                        }
                        if (pd->TimeIntegration != STEADY) {
                          if (pd->e[pg->imtrx][eqn] & T_MASS) {
                            dZi -= (1. + 2. * tt) * phi_j / dt * grad_b[p][ii][jj];
                          }
                        }
                        dZi *= at * lambda;
                        dbl sign = SGN(Z[ii][jj]);
                        dbl dZ = dZi * sign;

                        dbl d_scaling_grad_s = 0;
                        for (int w = 0; w < dim; w++) {
                          d_scaling_grad_s +=
                              2.0 * grad_b[w][ii][jj] * d_grad_b_dmesh[w][ii][jj][p][j];
                        }

                        dbl d_he = 0;
                        for (int q = 0; q < ei[pg->imtrx]->dof[eqn]; q++) {
                          dbl tmp = 0;
                          dbl dtmp = 0;
                          for (int w = 0; w < dim; w++) {
                            tmp += bf[eqn]->grad_phi[q][w] * bf[eqn]->grad_phi[q][w];
                            dtmp += 2.0 * bf[eqn]->d_grad_phi_dmesh[q][w][p][j] *
                                    bf[eqn]->grad_phi[q][w];
                          }
                          d_he += -0.5 * dtmp / pow(tmp, 3.0 / 2.0);
                        }

                        dbl dkdc = 0;
                        if (vn->shockcaptureModel == SC_YZBETA ||
                            vn->shockcaptureModel == YZBETA_ONE ||
                            vn->shockcaptureModel == YZBETA_MIXED) {
                          dkdc += 2.0 * d_he * he * Y_inv * fabs(Z[ii][jj]) / mag_b +
                                  Y_inv * dZ * he * he / mag_b;
                        }
                        if (vn->shockcaptureModel == SC_YZBETA ||
                            vn->shockcaptureModel == YZBETA_TWO ||
                            vn->shockcaptureModel == YZBETA_MIXED) {
                          dkdc += Y_inv * dZ * (1.0 / sqrt(scaling_grad_s[ii][jj])) * he / mag_b;
                          dkdc += Y_inv * fabs(Z[ii][jj]) * (1.0 / sqrt(scaling_grad_s[ii][jj])) *
                                  d_he / mag_b;
                          dkdc +=
                              Y_inv * fabs(Z[ii][jj]) *
                              (-0.5 * d_scaling_grad_s / pow(scaling_grad_s[ii][jj], 3.0 / 2.0)) *
                              he / mag_b;
                        }
                        if (vn->shockcaptureModel == SC_YZBETA ||
                            vn->shockcaptureModel == YZBETA_MIXED) {
                          dkdc *= 0.5;
                        }
                        dbl scale = kdc[ii][jj];
                        dbl dscale = dkdc;
                        if (supg > 0) {
                          dbl tmp = kdc[ii][jj];
                          dbl tau = supg_terms.supg_tau;
                          dbl dtau = supg_terms.d_supg_tau_dX[p][j];
                          scale = 1.0 / (sqrt(1.0 / (tmp * tmp) + 1.0 / (tau * tau)) + 1e-16);
                          dscale = 0.5 * scale * scale * scale * 2 *
                                   (dkdc / pow(kdc[ii][jj], 3) + dtau / pow(dtau, 3));
                        }
                        dbl diffusion_a = 0;
                        dbl diffusion_b = 0;
                        for (int r = 0; r < VIM; r++) {
                          diffusion_a += dscale * grad_b[r][ii][jj] * bf[eqn]->grad_phi[i][r];
                          diffusion_a +=
                              scale * d_grad_b_dmesh[r][ii][jj][p][j] * bf[eqn]->grad_phi[i][r];
                          diffusion_a +=
                              scale * grad_b[r][ii][jj] * bf[eqn]->d_grad_phi_dmesh[i][r][p][j];
                          diffusion_b += scale * grad_b[r][ii][jj] * bf[eqn]->grad_phi[i][r];
                        }
                        diffusion_a *= yzbeta_factor * det_J * wt * h3;
                        diffusion_b *=
                            yzbeta_factor * (d_det_J_dmesh_pj * h3 + det_J * dh3dmesh_pj) * wt;
                        diffusion = diffusion_a + diffusion_b;
                        diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
                      }
                    }

                    /*
                     * Source term...
                     */

                    source = 0.;

                    if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                      source_a = R_source;

                      source_a *= wt_func * (d_det_J_dmesh_pj * h3 + det_J * dh3dmesh_pj);

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

                      source = source_a + source_c;

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
                        advection = 0.;
                        advection_a = 0.;
                        if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                          if (DOUBLE_NONZERO(lambda)) {
                            for (int k = 0; k < VIM; k++) {
                              advection += -b[ii][k] * delta(p, k) * delta(jj, q);
                            }
                            advection += -d_a_dot_b_dG[p][q][ii][jj];
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
                          if (shock_is_yzbeta(vn->shockcaptureModel)) {
                            dbl dZi = 0;
                            for (int k = 0; k < VIM; k++) {
                              dZi += -b[ii][k] * delta(p, k) * delta(jj, q) * phi_j;
                            }
                            dZi -= phi_j * d_a_dot_b_dG[p][q][ii][jj];
                            dZi *= at * lambda;
                            dbl sign = SGN(Z[ii][jj]);
                            dbl dZ = dZi * sign;
                            dbl dkdc = 0;
                            if (vn->shockcaptureModel == SC_YZBETA ||
                                vn->shockcaptureModel == YZBETA_ONE ||
                                vn->shockcaptureModel == YZBETA_MIXED) {
                              dkdc += Y_inv * dZ * he * he / mag_b;
                            }
                            if (vn->shockcaptureModel == SC_YZBETA ||
                                vn->shockcaptureModel == YZBETA_TWO ||
                                vn->shockcaptureModel == YZBETA_MIXED) {
                              dkdc += Y_inv * dZ * (1 / sqrt(scaling_grad_s[ii][jj])) * he / mag_b;
                            }
                            if (vn->shockcaptureModel == SC_YZBETA ||
                                vn->shockcaptureModel == YZBETA_MIXED) {
                              dkdc *= 0.5;
                            }
                            dbl dscale = dkdc;
                            if (supg > 0) {
                              dbl tmp = kdc[ii][jj];
                              dbl tau = supg_terms.supg_tau;
                              dbl scale =
                                  1.0 / (sqrt(1.0 / (tmp * tmp) + 1.0 / (tau * tau)) + 1e-16);
                              dscale =
                                  0.5 * scale * scale * scale * 2 * (dkdc / pow(kdc[ii][jj], 3));
                            }
                            for (int r = 0; r < VIM; r++) {
                              diffusion += dscale * grad_b[r][ii][jj] * bf[eqn]->grad_phi[i][r];
                            }

                            diffusion *= yzbeta_factor * det_J * wt * h3;
                            diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
                          }
                        }

                        /*
                         * Source term...
                         */

                        source = 0.;

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

                      mass = b_dot[ii][jj];
                      mass *= d_lambda_dF[j];
                      mass *= wt_func * at * det_J * wt;
                      mass *= h3;
                      mass *= pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
                    }
                  }

                  advection = 0.;

                  if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                    if (d_lambda_dF[j] != 0.) {

                      advection += R_advection;
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

                    // Giesekus
                    if (alpha != 0.) {
                      source += s_dot_s[ii][jj] *
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
                  if (q >= p) {
                    var = v_s[mode][p][q];

                    if (pd->v[pg->imtrx][var]) {
                      pvar = upd->vp[pg->imtrx][var];
                      for (j = 0; j < ei[pg->imtrx]->dof[var]; j++) {
                        phi_j = bf[var]->phi[j];
                        mass = 0.;
                        if (pd->TimeIntegration != STEADY) {
                          if (pd->e[pg->imtrx][eqn] & T_MASS) {
                            mass = (1. + 2. * tt) * phi_j / dt * (double)delta(ii, p) *
                                   (double)delta(jj, q);
                            mass *= h3 * det_J;
                            mass *=
                                wt_func * at * lambda * wt * pd->etm[pg->imtrx][eqn][(LOG2_MASS)];
                          }
                        }

                        advection = 0.;

                        if (pd->e[pg->imtrx][eqn] & T_ADVECTION) {
                          if (DOUBLE_NONZERO(lambda)) {
                            if ((ii == p) && (jj == q)) {
                              for (r = 0; r < WIM; r++) {
                                advection += (v[r] - x_dot[r]) * bf[var]->grad_phi[j][r];
                              }
                            }

                            for (int k = 0; k < VIM; k++) {
                              advection -=
                                  phi_j *
                                  (delta(ii, q) * delta(k, p) | delta(ii, p) * delta(k, q)) *
                                  g[k][jj];
                            }
                            advection -= phi_j * d_a_dot_b_db[p][q][ii][jj];

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
                          if (shock_is_yzbeta(vn->shockcaptureModel)) {
                            dbl dZi = 0;
                            if (pd->TimeIntegration != STEADY) {
                              if (pd->e[pg->imtrx][eqn] & T_MASS) {
                                dZi = (1. + 2. * tt) * phi_j / dt * (double)delta(ii, p) *
                                      (double)delta(jj, q);
                              }
                            }
                            if ((ii == p) && (jj == q)) {
                              for (r = 0; r < WIM; r++) {
                                dZi += (v[r] - x_dot[r]) * bf[var]->grad_phi[j][r];
                              }
                            }

                            for (int k = 0; k < VIM; k++) {
                              dZi -= phi_j *
                                     (delta(ii, q) * delta(k, p) | delta(ii, p) * delta(k, q)) *
                                     g[k][jj];
                            }
                            dZi -= d_a_dot_b_db[p][q][ii][jj] * phi_j;
                            dZi *= at * lambda;
                            dZi += d_source_term_db[ii][jj][p][q] * phi_j;
                            dbl sign = SGN(Z[ii][jj]);
                            dbl dZ = dZi * sign;

                            dbl d_scaling_grad_s = 0;
                            for (int w = 0; w < dim; w++) {
                              d_scaling_grad_s += 2.0 * grad_b[w][ii][jj] *
                                                  bf[var]->grad_phi[j][w] * delta(ii, p) *
                                                  delta(jj, q);
                            }

                            dbl tmp = 0;
                            dbl d_mag_b = 0;
                            tmp += b[p][q] * bf[var]->phi[j];
                            if (p != q) {
                              tmp += b[p][q] * bf[var]->phi[j];
                            }
                            d_mag_b = tmp / mag_b;

                            dbl dkdc = 0;
                            if (vn->shockcaptureModel == SC_YZBETA ||
                                vn->shockcaptureModel == YZBETA_ONE ||
                                vn->shockcaptureModel == YZBETA_MIXED) {
                              dkdc +=
                                  -Y_inv * fabs(Z[ii][jj]) * d_mag_b * he * he / (mag_b * mag_b) +
                                  Y_inv * dZ * he * he / mag_b;
                            }
                            if (vn->shockcaptureModel == SC_YZBETA ||
                                vn->shockcaptureModel == YZBETA_TWO ||
                                vn->shockcaptureModel == YZBETA_MIXED) {
                              dkdc +=
                                  Y_inv * dZ * (1.0 / sqrt(scaling_grad_s[ii][jj])) * he / mag_b;
                              dkdc += -Y_inv * fabs(Z[ii][jj]) *
                                      (1.0 / sqrt(scaling_grad_s[ii][jj])) * d_mag_b * he /
                                      (mag_b * mag_b);
                              dkdc += Y_inv * fabs(Z[ii][jj]) *
                                      (-0.5 * d_scaling_grad_s /
                                       pow(scaling_grad_s[ii][jj], 3.0 / 2.0)) *
                                      he / mag_b;
                            }
                            dkdc *= 0.5;
                            dbl scale = kdc[ii][jj];
                            dbl dscale = dkdc;
                            if (supg > 0) {
                              dbl tmp = kdc[ii][jj];
                              dbl tau = supg_terms.supg_tau;
                              scale = 1.0 / (sqrt(1.0 / (tmp * tmp) + 1.0 / (tau * tau)) + 1e-16);
                              dscale =
                                  0.5 * scale * scale * scale * 2 * (dkdc / pow(kdc[ii][jj], 3));
                            }
                            for (int r = 0; r < VIM; r++) {
                              diffusion += dscale * grad_b[r][ii][jj] * bf[eqn]->grad_phi[i][r];
                              diffusion += scale * delta(p, ii) * delta(q, jj) *
                                           bf[var]->grad_phi[j][r] * bf[eqn]->grad_phi[i][r];
                            }
                          }

                          diffusion *= yzbeta_factor * det_J * wt * h3;
                          diffusion *= pd->etm[pg->imtrx][eqn][(LOG2_DIFFUSION)];
                        }

                        /*
                         * Source term...
                         */

                        source = 0.;

                        if (pd->e[pg->imtrx][eqn] & T_SOURCE) {
                          source = d_source_term_db[ii][jj][p][q];
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
      }
    } /* End Assemble Jacobian */
  }   /* End loop over modes */

  return (status);
}