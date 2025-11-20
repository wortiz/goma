#include "ad/beer_belly.h"
#include "ad/structs.h"

extern "C" {
#include "el_elm.h"
#include "el_elm_info.h"
#include "el_geom.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "mm_eh.h"
#include "mm_mp.h"
#include "mm_mp_structs.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "std.h"
}

int ad_beer_belly(void) {
  if (ad_fv == NULL) {
    ad_fv = new AD_Field_Variables();
  }
  int status = 0, i, j, k, dim, pdim, mdof, index, node, si;
  int DeformingMesh, ShapeVar;
  struct Basis_Functions *MapBf;
  int imtrx = upd->matrix_index[pd->ShapeVar];

  static int is_initialized = FALSE;
  static int elem_blk_id_save = -123;

  dim = ei[imtrx]->ielem_dim;
  pdim = pd->Num_Dim;
  int elem_type = ei[imtrx]->ielem_type;
  int elem_shape = type2shape(elem_type);

  ShapeVar = pd->ShapeVar;

  /* If this is a shell element, it may be a deforming mesh
   * even if there are no mesh equations on the shell block.
   * The ei[imtrx]->deforming_mesh flag is TRUE for shell elements when
   * there are mesh equations active on either the shell block or
   * any neighboring bulk block.
   */

  if (pd->gv[MESH_DISPLACEMENT1]) {
    DeformingMesh = ei[upd->matrix_index[MESH_DISPLACEMENT1]]->deforming_mesh;
  } else {
    DeformingMesh = ei[imtrx]->deforming_mesh;
  }

  if ((si = in_list(pd->IntegrationMap, 0, Num_Interpolations, Unique_Interpolations)) == -1) {
    GOMA_EH(GOMA_ERROR, "Seems to be a problem finding the IntegrationMap interpolation.");
  }
  MapBf = bfd[si];

  mdof = ei[imtrx]->dof[ShapeVar];

  if (MapBf->interpolation == I_N1) {
    mdof = MapBf->shape_dof;
  }

  /*
   * For every type "t" of unique basis function used in this problem,
   * initialize appropriate arrays...
   */

  if (ei[imtrx]->elem_blk_id != elem_blk_id_save) {
    is_initialized = FALSE;
  }

  if (!is_initialized) {
    is_initialized = TRUE;
    elem_blk_id_save = ei[imtrx]->elem_blk_id;
  }

  /*
   * For convenience, while we are here, interpolate to find physical space
   * location using the mesh basis function.
   *
   * Generally, the other basis functions will give various different estimates
   * for position, depending on whether the shapes are mapped sub/iso/super
   * parametrically...
   */
  for (i = 0; i < VIM; i++) {
    ad_fv->x[i] = 0.;
  }

  /*
   * NOTE: pdim is the number of coordinates, which may differ from
   * the element dimension (dim), as for shell elements!
   */
  for (i = 0; i < pdim; i++) {
    if (DeformingMesh) {
      for (k = 0; k < ei[upd->matrix_index[R_MESH1]]->dof[R_MESH1]; k++) {
        node = ei[upd->matrix_index[R_MESH1]]->dof_list[R_MESH1][k];

        index = Proc_Elem_Connect[Proc_Connect_Ptr[ei[upd->matrix_index[R_MESH1]]->ielem] + node];

        ad_fv->x[i] += (Coor[i][index] + ADType(ad_fv->total_ad_variables,
                                                ad_fv->offset[R_MESH1 + i] + k, *esp->d[i][k])) *
                       bf[R_MESH1]->phi[k];
      }
    } else {
      for (k = 0; k < mdof; k++) {
        node = MapBf->interpolation == I_N1 ? k : ei[imtrx]->dof_list[ShapeVar][k];

        index = Proc_Elem_Connect[Proc_Connect_Ptr[ei[imtrx]->ielem] + node];

        ad_fv->x[i] += Coor[i][index] * MapBf->phi[k];
      }
    }
  }

  /*
   * Elemental Jacobian is now affected by mesh displacement of nodes
   * from their initial nodal point coordinates...
   */

  for (i = 0; i < dim; i++) {
    for (j = 0; j < pdim; j++) {
      ad_fv->J[i][j] = 0.0;
      if (DeformingMesh) {
        for (k = 0; k < ei[upd->matrix_index[R_MESH1]]->dof[R_MESH1]; k++) {
          node = ei[upd->matrix_index[R_MESH1]]->dof_list[R_MESH1][k];

          index = Proc_Elem_Connect[Proc_Connect_Ptr[ei[upd->matrix_index[R_MESH1]]->ielem] + node];

          ad_fv->J[i][j] +=
              (Coor[j][index] +
               ADType(ad_fv->total_ad_variables, ad_fv->offset[R_MESH1 + j] + k, *esp->d[j][k])) *
              bf[R_MESH1]->dphidxi[k][i];
        }
      } else {
        for (k = 0; k < mdof; k++) {
          node = MapBf->interpolation == I_N1 ? k : ei[imtrx]->dof_list[ShapeVar][k];
          index = Proc_Elem_Connect[Proc_Connect_Ptr[ei[imtrx]->ielem] + node];
          ad_fv->J[i][j] += Coor[j][index] * bf[ShapeVar]->dphidxi[k][i];
        }
      }
    }
  }

  if (elem_shape == SHELL || elem_shape == TRISHELL || (mp->ehl_integration_kind == SIK_S)) {
    dim++;
    for (j = 0; j < pdim; j++) {
      ad_fv->J[pd->Num_Dim - 1][j] = MapBf->J[pd->Num_Dim - 1][j] = (j + 1) * 1.0;
    }

    /*Real Quick check on Jacobian to make sure this arbitrary assignment
     *didn't screw things up. Note that the detJ in the shell case can be
     *negative, but it is important to point out that we are not using it for
     *for integration, but only as a crutch for inversion of J */
    if (pd->Num_Dim == 3) {
      ad_fv->detJ =
          ad_fv->J[0][0] * (ad_fv->J[1][1] * ad_fv->J[2][2] - ad_fv->J[1][2] * ad_fv->J[2][1]) -
          ad_fv->J[0][1] * (ad_fv->J[1][0] * ad_fv->J[2][2] - ad_fv->J[2][0] * ad_fv->J[1][2]) +
          ad_fv->J[0][2] * (ad_fv->J[1][0] * ad_fv->J[2][1] - ad_fv->J[2][0] * ad_fv->J[1][1]);
    }
    if (pd->Num_Dim == 2) {
      ad_fv->detJ = ad_fv->J[0][0] * ad_fv->J[1][1] - ad_fv->J[0][1] * ad_fv->J[1][0];
    }

    if (fabs(ad_fv->detJ) < 1.e-10) {
      zero_detJ = TRUE;
#ifdef PARALLEL
      fprintf(stderr, "\nP_%d: Uh-oh, detJ =  %e\n", ProcID, fabs(ad_fv->detJ.val()));
#else
      fprintf(stderr, "\n Uh-oh, detJ =  %e\n", fabs(ad_fv->detJ));
#endif
      return (2);
    }
  }

  /* Compute inverse of Jacobian for only the MapBf right now */

  /*
   * Wiggly mesh derivatives..
   */
  switch (dim) {
  case 1:
    GOMA_EH(GOMA_ERROR, "dim = 1 not implemented in ad_beer_belly");
    break;

  case 2:
    dim = ei[pg->imtrx]->ielem_dim;
    ad_fv->detJ = ad_fv->J[0][0] * ad_fv->J[1][1] - ad_fv->J[0][1] * ad_fv->J[1][0];

    ad_fv->B[0][0] = ad_fv->J[1][1] / ad_fv->detJ;
    ad_fv->B[0][1] = -ad_fv->J[0][1] / ad_fv->detJ;
    ad_fv->B[1][0] = -ad_fv->J[1][0] / ad_fv->detJ;
    ad_fv->B[1][1] = ad_fv->J[0][0] / ad_fv->detJ;

    break;

  case 3:

    /* Now that we are here, reset dim for the shell case */
    dim = ei[imtrx]->ielem_dim;

    ad_fv->detJ =
        ad_fv->J[0][0] * (ad_fv->J[1][1] * ad_fv->J[2][2] - ad_fv->J[1][2] * ad_fv->J[2][1]) -
        ad_fv->J[0][1] * (ad_fv->J[1][0] * ad_fv->J[2][2] - ad_fv->J[2][0] * ad_fv->J[1][2]) +
        ad_fv->J[0][2] * (ad_fv->J[1][0] * ad_fv->J[2][1] - ad_fv->J[2][0] * ad_fv->J[1][1]);

    ad_fv->B[0][0] =
        (ad_fv->J[1][1] * ad_fv->J[2][2] - ad_fv->J[2][1] * ad_fv->J[1][2]) / (ad_fv->detJ);

    ad_fv->B[0][1] =
        -(ad_fv->J[0][1] * ad_fv->J[2][2] - ad_fv->J[2][1] * ad_fv->J[0][2]) / (ad_fv->detJ);

    ad_fv->B[0][2] =
        (ad_fv->J[0][1] * ad_fv->J[1][2] - ad_fv->J[1][1] * ad_fv->J[0][2]) / (ad_fv->detJ);

    ad_fv->B[1][0] =
        -(ad_fv->J[1][0] * ad_fv->J[2][2] - ad_fv->J[2][0] * ad_fv->J[1][2]) / (ad_fv->detJ);

    ad_fv->B[1][1] =
        (ad_fv->J[0][0] * ad_fv->J[2][2] - ad_fv->J[2][0] * ad_fv->J[0][2]) / (ad_fv->detJ);

    ad_fv->B[1][2] =
        -(ad_fv->J[0][0] * ad_fv->J[1][2] - ad_fv->J[1][0] * ad_fv->J[0][2]) / (ad_fv->detJ);

    ad_fv->B[2][0] =
        (ad_fv->J[1][0] * ad_fv->J[2][1] - ad_fv->J[1][1] * ad_fv->J[2][0]) / (ad_fv->detJ);

    ad_fv->B[2][1] =
        -(ad_fv->J[0][0] * ad_fv->J[2][1] - ad_fv->J[2][0] * ad_fv->J[0][1]) / (ad_fv->detJ);

    ad_fv->B[2][2] =
        (ad_fv->J[0][0] * ad_fv->J[1][1] - ad_fv->J[1][0] * ad_fv->J[0][1]) / (ad_fv->detJ);

    break;

  default:
    GOMA_EH(GOMA_ERROR, "Bad dim.");
    break;
  }
  return (status);
}