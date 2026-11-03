/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) 2022 Goma Developers, National Technology & Engineering   *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

/*
 *$Id: mm_as_alloc.c,v 5.19 2010-04-07 22:27:00 prschun Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "el_elm.h"  /* Has shape, element type stuff for bf_init */
#include "el_geom.h" /* Has info I'd like to replicate into the */
#include "rf_allo.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_io.h"
#include "std.h"
/* Problem_Description structure... */
#include "el_elm_info.h"
#include "exo_struct.h"
#include "mm_as.h"
#include "mm_as_alloc.h"
#include "mm_as_const.h"
#include "mm_as_structs.h"
#include "mm_mp_const.h"
#include "mm_mp_structs.h"
#include "rd_mesh.h"
#include "rf_vars_const.h"

#define GOMA_MM_AS_ALLOC_C

static void init_Viscoelastic_Nonmodal(struct Viscoelastic_Nonmodal *);

static void init_Viscoelastic_Constitutive(struct Viscoelastic_Constitutive *);

static void init_Generalized_Newtonian(GEN_NEWT_STRUCT *);

static void init_Elastic_Constitutive(struct Elastic_Constitutive *);

static void init_Viscoplastic_Constitutive(struct Viscoplastic_Constitutive *);

static int shape_list(Exo_DB *);

// The FEM working pointers formerly defined here now live in goma_ctx->fem.

/*
 * This is where these are declared. The pointer is loaded with the
 * address of a double zero that will be used when element degree of freedom
 * pointers need to be assigned to point to something harmless.
 */

dbl zero = 0.0;

/*
 * In this memory allocation routine, "pd", etc. have one level of indirection
 * greater than their usual meaning. The main driver "main" calls these
 * routines to make sure that "pd", "ei" point to places with enough
 * space to hold the relevent information.
 */

#include "mm_eh.h"

/****************************************************************************/

static void Element_Indices_free(struct Element_Indices *ei_ptr) {
  if (ei_ptr == NULL) {
    return;
  }
  free(ei_ptr->Num_Lvdesc_Per_Var_Type);
  free(ei_ptr->Lvdesc_First_Var_Type);
  free(ei_ptr->Lvdesc_to_ledof);
  free(ei_ptr->Lvdesc_to_lvdof);
  free(ei_ptr->Lvdesc_to_Var_Type);
  free(ei_ptr->Lvdesc_to_MatID);
  free(ei_ptr->Lvdesc_Numdof);
  free(ei_ptr->Lvdesc_to_Gnn);
  free(ei_ptr->Lvdesc_to_Gun);
  free(ei_ptr->Lvdesc_to_MFSubvar);
  free(ei_ptr->Lvdesc_to_Lnn);
  free(ei_ptr->Lvdesc_Lnn_to_Offset);
  free(ei_ptr->Lvdesc_Lnn_to_lvdof);
  free(ei_ptr->lvdof_to_lvdesc);
  free(ei_ptr->lvdof_to_lvdesc_dof);
  free(ei_ptr->Lvdesc_Lnn_Numdof);
  free(ei_ptr->Lvdesc_vd_ptr);
  free(ei_ptr->owningElementForColVar);
  free(ei_ptr->dof);
  free(ei_ptr->dof_ext);
  free(ei_ptr->Baby_Dolphin);
  free(ei_ptr->dof_list);
  free(ei_ptr->gnn_list);
  free(ei_ptr->gun_list);
  free(ei_ptr->ln_to_dof);
  free(ei_ptr->ln_to_first_dof);
  free(ei_ptr->lvdof_to_row_lvdof);
  free(ei_ptr->lvdof_to_ledof);
  free(ei_ptr->owned_ledof);
  free(ei_ptr->ieqn_ledof);
  free(ei_ptr->matID_ledof);
  free(ei_ptr->active_interp_ledof);
  free(ei_ptr->MFsubvar_Offset);
  free(ei_ptr);
}

static void Element_Stiffness_Pointers_free(struct Element_Stiffness_Pointers *e) {
  if (e == NULL) {
    return;
  }
  /* a0 points into the global Jacobian and is not owned */
  free(e->T);
  free(e->v);
  free(e->v_star);
  free(e->d);
  free(e->d_rs);
  free(e->c);
  free(e->P);
  free(e->P_star);
  free(e->S);
  free(e->G);
  free(e->V);
  free(e->qs);
  free(e->F);
  free(e->SH);
  free(e->Enorm);
  free(e->H);
  free(e->pv);
  free(e->p_liq);
  free(e->p_gas);
  free(e->porosity);
  free(e->vd);
  free(e->vlambda);
  free(e->nn);
  free(e->lm);
  free(e->ext_v);
  free(e->E_field);
  free(e->n);
  free(e->sh_K);
  free(e->sh_K2);
  free(e->sh_tens);
  free(e->sh_x);
  free(e->sh_y);
  free(e->sh_u);
  free(e->sh_ang);
  free(e->div_s_v);
  free(e->curv);
  free(e->grad_v_dot_n);
  free(e->n_dot_curl_s_v);
  free(e->pF);
  free(e->sh_J);
  free(e->sh_Kd);
  free(e->apr);
  free(e->api);
  free(e->ars);
  free(e->epr);
  free(e->epi);
  free(e->sink_mass);
  free(e->sh_bv);
  free(e->sh_p);
  free(e->lubp);
  free(e->lubp_2);
  free(e->sh_fp);
  free(e->sh_fh);
  free(e->sh_pc);
  free(e->sh_sat_closed);
  free(e->sh_p_open);
  free(e->sh_p_open_2);
  free(e->sh_t);
  free(e->sh_dh);
  free(e->sh_l_curv);
  free(e->sh_l_curv_2);
  free(e->sh_sat_gasn);
  free(e->sh_shear_top);
  free(e->sh_shear_bot);
  free(e->sh_cross_shear);
  free(e->max_strain);
  free(e->cur_strain);
  free(e->poynt);
  free(e->tfmp_pres);
  free(e->tfmp_sat);
  free(e->moment);
  free(e->rho);
  free(e->restime);
  free(e->em_er);
  free(e->em_ei);
  free(e->em_hr);
  free(e->em_hi);
  free(e->sh_sat_1);
  free(e->sh_sat_2);
  free(e->sh_sat_3);
  free(e->eddy_nu);
  free(e->turb_k);
  free(e->turb_omega);
  free(e->film_height);
  free(e);
}

static void Porous_Media_Variables_Hysteresis_free(struct Porous_Media_Variables_Hysteresis *h) {
  if (h == NULL) {
    return;
  }
  for (size_t i = 0; i < sizeof(h->curve_type) / sizeof(h->curve_type[0]); i++) {
    free(h->curve_type[i]);
    free(h->curve_type_old[i]);
    free(h->curve_switch[i]);
    free(h->num_switch[i]);
    free(h->sat_switch[i]);
    free(h->cap_pres_switch[i]);
    free(h->sat_min_imbibe[i]);
    free(h->sat_max_drain[i]);
  }
  free(h);
}

/*
 * fem_structures_free():
 *     Free the structures allocated by the *_alloc() routines in this file and
 *     by assembly_alloc(). Installed as the GomaFem release hook by pd_alloc()
 *     so that goma_context_release() can call it.
 *
 *     Not yet released: memory hung off these structures while reading input or
 *     solving (material property tables, augmenting/continuation/hunting
 *     conditions, level set and phase function data, rotation vectors).
 */
void fem_structures_free(GomaFem *fem) {
  int mn, mode;

  if (fem->ei != NULL && fem->upd != NULL) {
    for (int imtrx = 0; imtrx < fem->upd->Total_Num_Matrices; imtrx++) {
      Element_Indices_free(fem->ei[imtrx]);
    }
  }
  free(fem->ei);
  if (fem->ei_related != NULL) {
    for (mn = 0; mn < MAX_ELEMENT_INDICES_RELATED; mn++) {
      Element_Indices_free(fem->ei_related[mn]);
    }
  }
  free(fem->ei_related);

  Element_Stiffness_Pointers_free(fem->esp);
  free(fem->esp_old);
  free(fem->esp_dot);
  free(fem->esp_dbl_dot);
  free(fem->evp);
  free(fem->eqm);
  free(fem->lec);
  free(fem->af);

  /* bf, bfi and bfex hold pointers into bfd */
  if (fem->bfd != NULL) {
    for (int b = 0; b < goma_ctx->problem.num_basis_functions; b++) {
      if (fem->bfd[b] != NULL) {
        free(fem->bfd[b]->Var_Type_MatID);
        free(fem->bfd[b]);
      }
    }
  }
  free(fem->bfd);
  free(fem->bfi);
  free(fem->bf);
  free(fem->bfex);

  free(fem->fv);
  free(fem->fv_sens);
  free(fem->fv_old);
  free(fem->fv_dot);
  free(fem->fv_dot_old);
  free(fem->fv_dot_dot);
  free(fem->fv_dot_dot_old);
  free(fem->efv);
  free(fem->pmv);
  free(fem->pmv_old);
  free(fem->pmv_ml);
  Porous_Media_Variables_Hysteresis_free(fem->pmv_hyst);
  free(fem->lub_aux);
  free(fem->lub_aux_old);
  free(fem->stab);

  free(fem->ve); /* rf_solve may already have freed this and cleared it */
  if (fem->ve_glob != NULL) {
    for (mn = 0; mn < MAX_NUMBER_MATLS; mn++) {
      for (mode = 0; mode < MAX_MODES; mode++) {
        if (fem->ve_glob[mn][mode] != NULL) {
          free(fem->ve_glob[mn][mode]->gn);
          free(fem->ve_glob[mn][mode]->time_const_st);
          free(fem->ve_glob[mn][mode]);
        }
      }
    }
  }
  free(fem->ve_glob);

  for (mn = 0; mn < MAX_NUMBER_MATLS; mn++) {
    if (fem->evpl_glob != NULL && fem->evpl_glob[mn] != NULL) {
      struct Viscoplastic_Constitutive *v = fem->evpl_glob[mn];
      free(v->F_vp_glob);
      free(v->F_vp_old_glob);
      free(v->TT_glob);
      free(v->TT_old_glob);
      free(v->dTT_dx_glob);
      free(v->dTT_dx_old_glob);
      free(v->dTT_dc_glob);
      free(v->dTT_dc_old_glob);
      free(v);
    }
    if (fem->vn_glob != NULL) {
      free(fem->vn_glob[mn]);
    }
    if (fem->elc_glob != NULL) {
      free(fem->elc_glob[mn]);
    }
    if (fem->elc_rs_glob != NULL) {
      free(fem->elc_rs_glob[mn]);
    }
    if (fem->gn_glob != NULL) {
      free(fem->gn_glob[mn]);
    }
    if (fem->cr_glob != NULL) {
      free(fem->cr_glob[mn]);
    }
    if (fem->mp_glob != NULL) {
      free(fem->mp_glob[mn]);
    }
    if (fem->pd_glob != NULL) {
      free(fem->pd_glob[mn]);
    }
  }
  free(fem->evpl_glob);
  free(fem->vn_glob);
  free(fem->elc_glob);
  free(fem->elc_rs_glob);
  free(fem->gn_glob);
  free(fem->cr_glob);
  free(fem->mp_glob);
  free(fem->mp_old);
  free(fem->pd_glob);

  free(fem->tran);
  free(fem->libio);
  free(fem->eigen);
  free(fem->cont);
  free(fem->loca_in);

  free(fem->pg);
  free(fem->upd);
}

/****************************************************************************/
/****************************************************************************/
/****************************************************************************/

int pd_alloc(void)

/******************************************************************
 *
 * pd_alloc():
 *     Assign space for and initialize the uniform problem
 * description structure and the problem description structures.
 ******************************************************************/
{
  int status = 0;
  int mn;
  static char yo[] = "pd_alloc";
  goma_ctx->fem.release = fem_structures_free;
  goma_ctx->fem.upd = alloc_struct_1(struct Uniform_Problem_Description, 1);
  goma_ctx->fem.upd->Species_Var_Type = SPECIES_UNDEFINED_FORM;

  goma_ctx->fem.pd_glob = (struct Problem_Description **)alloc_ptr_1(MAX_NUMBER_MATLS);
  for (mn = 0; mn < MAX_NUMBER_MATLS; mn++) {
    goma_ctx->fem.pd_glob[mn] = alloc_struct_1(struct Problem_Description, 1);
    goma_ctx->fem.pd_glob[mn]->Species_Var_Type = SPECIES_UNDEFINED_FORM;
  }

  goma_ctx->fem.pg = alloc_struct_1(struct Problem_Graph, 1);

  int imtrx;
  for (imtrx = 0; imtrx < MAX_NUM_MATRICES; imtrx++) {
    goma_ctx->fem.pg->time_step_control_disabled[imtrx] = 0;
  }

  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Problem_Description @ %p has %ld bytes", yo, (void *)goma_ctx->fem.pd_glob,
            (long int)sizeof(struct Problem_Description));
  }
  if (goma_ctx->fem.upd == NULL) {
    status = -1;
    GOMA_EH(status, "Hay un problema conseguiendo la memoria para Uniform_Problem_Description");
  }
  if (goma_ctx->fem.pd_glob == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Problem_Description");
  }
  if (goma_ctx->fem.pg == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Problem_Graph");
  }

  return (status);
}
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/

int efv_alloc(void) {
  int sz;
  int status = 0;
  static char yo[] = "efv_alloc";

  /*
   * External_Field_Variables___________________________________________________
   */

  sz = sizeof(struct External_Field_Variables);
  goma_ctx->fem.efv = (struct External_Field_Variables *)array_alloc(1, 1, sz);
  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: External_Field_Variables @ %p has %d bytes", yo, (void *)goma_ctx->fem.efv,
            sz);
  }

  if (goma_ctx->fem.efv == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for External_Field_Variables");
  }

  memset(goma_ctx->fem.efv, 0, sz);

  return (status);
}

/***************************************************************************/
/***************************************************************************/
/***************************************************************************/

int mp_alloc(void)

/***********************************************************************
 *
 * mp_alloc()
 *
 *  Allocate Material_Properties structures and also allocate
 *  substructures underneath the main structure (there are none as
 *  of yet).
 *
 *  Right now, we allocate a fixed number of material property
 *  structures given by the constant, MAX_NUMBER_MATLS, plus an
 *  additional Material_Property structure, mp_old, which I don't know
 *  the reason for.
 *
 *  Return
 *       0 -> Everything went well
 *       fatal error -> Can't malloc enough memory
 ***********************************************************************/
{
  int mn;
  static char yo[] = "mp_alloc";

  goma_ctx->fem.mp_glob = (MATRL_PROP_STRUCT **)alloc_ptr_1(MAX_NUMBER_MATLS);
  if (goma_ctx->fem.mp_glob == NULL) {
    GOMA_EH(-1, "Problem getting memory for Material_Properties");
  }

  for (mn = 0; mn < MAX_NUMBER_MATLS; mn++) {
    goma_ctx->fem.mp_glob[mn] =
        (MATRL_PROP_STRUCT *)alloc_void_struct_1(sizeof(MATRL_PROP_STRUCT), 1);
  }
  goma_ctx->fem.mp_old = (MATRL_PROP_STRUCT *)alloc_void_struct_1(sizeof(MATRL_PROP_STRUCT), 1);

  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Material_Properties @ %p has %lu bytes", yo, (void *)goma_ctx->fem.mp,
            (long unsigned int)sizeof(MATRL_PROP_STRUCT));
  }
  return 0;
}
/***************************************************************************/
/***************************************************************************/
/***************************************************************************/

int cr_alloc(void)

/******************************************************************
 *
 * cr_alloc:
 *
 *  Initializes the global structure, cr_glob.
 ******************************************************************/
{
  int sz;
  int mn;
  int status;
  static char yo[] = "cr_alloc";

  status = 0;

  /*
   * Constitutive_Relations____________________________________________________
   */

  sz = sizeof(struct Constitutive_Relations *);
  goma_ctx->fem.cr_glob = (struct Constitutive_Relations **)array_alloc(1, MAX_NUMBER_MATLS, sz);

  sz = sizeof(struct Constitutive_Relations);

  for (mn = 0; mn < MAX_NUMBER_MATLS; mn++) {
    goma_ctx->fem.cr_glob[mn] = (struct Constitutive_Relations *)array_alloc(1, 1, sz);
    memset(goma_ctx->fem.cr_glob[mn], 0, sz);
  }

  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Constitutive_Relations @ %p has %d bytes", yo, (void *)goma_ctx->fem.cr,
            sz);
  }

  if (goma_ctx->fem.cr_glob == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Constitutive_Relations");
  }

  return (status);
}
/***************************************************************************/
/***************************************************************************/
/***************************************************************************/

int gn_alloc(void) {
  int sz;
  int mn;
  int status;
  static char yo[] = "gn_alloc";

  status = 0;

  /*
   * Generalized_Newtonian______________________________________________________
   */

  sz = sizeof(struct Generalized_Newtonian *);
  goma_ctx->fem.gn_glob = (struct Generalized_Newtonian **)array_alloc(1, MAX_NUMBER_MATLS, sz);

  if (goma_ctx->fem.gn_glob == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Generalized_Newtonian");
  }

  sz = sizeof(struct Generalized_Newtonian);

  for (mn = 0; mn < MAX_NUMBER_MATLS; mn++) {
    goma_ctx->fem.gn_glob[mn] = (struct Generalized_Newtonian *)array_alloc(1, 1, sz);

    if (goma_ctx->fem.gn_glob[mn] == NULL) {
      status = -1;
      GOMA_EH(status, "Problem getting memory for Generalized_Newtonian material");
    }
    memset(goma_ctx->fem.gn_glob[mn], 0, sz);
  }

  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Generalized_Newtonian @ %p has %d bytes", yo, (void *)goma_ctx->fem.gn,
            sz);
  }

  return (status);
}

int ve_alloc(void) {
  int sz;
  int mn;
  int mode;
  int status;
  static char yo[] = "ve_alloc";

  status = 0;

  /*
   * Viscoelastic_Constitutive_________________________________________________
   */

  sz = sizeof(struct Viscoelastic_Nonmodal *);
  goma_ctx->fem.vn_glob = (struct Viscoelastic_Nonmodal **)smalloc(MAX_NUMBER_MATLS * sz);

  sz = sizeof(struct Viscoelastic_Nonmodal);

  for (mn = 0; mn < MAX_NUMBER_MATLS; mn++) {
    goma_ctx->fem.vn_glob[mn] = (struct Viscoelastic_Nonmodal *)smalloc(sz);
    init_Viscoelastic_Nonmodal(goma_ctx->fem.vn_glob[mn]);
  }

  sz = sizeof(struct Viscoelastic_Constitutive *);
  goma_ctx->fem.ve = (struct Viscoelastic_Constitutive **)smalloc(MAX_MODES * sz);

  sz = sizeof(struct Viscoelastic_Constitutive *);
  goma_ctx->fem.ve_glob =
      (struct Viscoelastic_Constitutive ***)array_alloc(2, MAX_NUMBER_MATLS, MAX_MODES, sz);

  sz = sizeof(struct Viscoelastic_Constitutive);

  for (mn = 0; mn < MAX_NUMBER_MATLS; mn++) {
    for (mode = 0; mode < MAX_MODES; mode++) {
      goma_ctx->fem.ve_glob[mn][mode] = (struct Viscoelastic_Constitutive *)smalloc(sz);
      init_Viscoelastic_Constitutive(goma_ctx->fem.ve_glob[mn][mode]);
    }
  }

  for (mn = 0; mn < MAX_NUMBER_MATLS; mn++) {
    for (mode = 0; mode < MAX_MODES; mode++) {
      goma_ctx->fem.ve_glob[mn][mode]->gn = alloc_struct_1(GEN_NEWT_STRUCT, 1);
      goma_ctx->fem.ve_glob[mn][mode]->time_const_st = alloc_struct_1(POLYMER_TIME_CONST_STRUCT, 1);
      init_Generalized_Newtonian(goma_ctx->fem.ve_glob[mn][mode]->gn);
    }
  }

  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Viscoelastic_Constitutive @ %p has %d bytes", yo,
            (void *)goma_ctx->fem.ve_glob, sz);
  }

  if (goma_ctx->fem.ve_glob == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Viscoelastic_Constitutive");
  }

  return (status);
}
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/

int elc_alloc(void) {
  int sz;
  int mn;
  int status;
  static char yo[] = "elc_alloc";

  status = 0;

  /*
   * Elastic_Constitutive______________________________________________________
   */

  sz = sizeof(struct Elastic_Constitutive *);
  goma_ctx->fem.elc_glob = (struct Elastic_Constitutive **)smalloc(MAX_NUMBER_MATLS * sz);

  sz = sizeof(struct Elastic_Constitutive);
  for (mn = 0; mn < MAX_NUMBER_MATLS; mn++) {
    goma_ctx->fem.elc_glob[mn] = (struct Elastic_Constitutive *)smalloc(sz);
    memset(goma_ctx->fem.elc_glob[mn], 0, sz);
    init_Elastic_Constitutive(goma_ctx->fem.elc_glob[mn]);
  }

  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Elastic Constitutive @ %p has %d bytes", yo, (void *)goma_ctx->fem.gn_glob,
            sz);
  }

  if (goma_ctx->fem.elc_glob == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Elastic_Constitutive");
  }

  return (status);
}
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/

int evp_alloc(void) {
  int sz;
  int mn;
  int status;
  static char yo[] = "evp_alloc";

  status = 0;

  /*
   * Viscoplastic_Constitutive_________________________________________________
   */

  sz = sizeof(struct Viscoplastic_Constitutive *);
  goma_ctx->fem.evpl_glob = (struct Viscoplastic_Constitutive **)smalloc(MAX_NUMBER_MATLS * sz);

  sz = sizeof(struct Viscoplastic_Constitutive);
  for (mn = 0; mn < MAX_NUMBER_MATLS; mn++) {
    goma_ctx->fem.evpl_glob[mn] = (struct Viscoplastic_Constitutive *)smalloc(sz);
    init_Viscoplastic_Constitutive(goma_ctx->fem.evpl_glob[mn]);
  }

  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Viscoplastic Constitutive @ %p has %d bytes", yo,
            (void *)goma_ctx->fem.gn_glob, sz);
  }

  if (goma_ctx->fem.evpl_glob == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Viscoplastic_Constitutive");
  }

  return (status);
}
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/

int evp_tensor_alloc(Exo_DB *exo) {
  int status;
  int ielem_type, ip_total, ielem0, mn;
  int i, ip, a, b, c, k, w;
  status = 0;

  /*
   * Viscoplastic global tensors_______________________________________________
   */

  for (mn = 0; mn < goma_ctx->fem.upd->Num_Mat; mn++) {
    /*Don't bother with allocation if we are not solving an EVP model */
    if (goma_ctx->fem.evpl_glob[mn]->ConstitutiveEquation == EVP_HYPER) {
      goma_ctx->fem.pd_glob[mn]->Num_Dim = goma_ctx->mesh.num_dim; /* from "el_geom.h" */
      ielem0 = exo->eb_ptr[mn];
      ielem_type = Elem_Type(exo, ielem0);     /* element type */
      ip_total = elem_info(NQUAD, ielem_type); /* number of */
      /* quadrature points */

      if (goma_ctx->fem.evpl_glob[mn] == NULL) {
        status = -1;
        GOMA_EH(status, "Problem getting memory for Viscoplastic_Constitutive");
      }

      goma_ctx->fem.evpl_glob[mn]->F_vp_glob =
          (dbl ****)array_alloc(4, exo->num_elems, ip_total, DIM, DIM, sizeof(dbl));
      goma_ctx->fem.evpl_glob[mn]->F_vp_old_glob =
          (dbl ****)array_alloc(4, exo->num_elems, ip_total, DIM, DIM, sizeof(dbl));
      goma_ctx->fem.evpl_glob[mn]->TT_glob =
          (dbl ****)array_alloc(4, exo->num_elems, ip_total, DIM, DIM, sizeof(dbl));
      goma_ctx->fem.evpl_glob[mn]->TT_old_glob =
          (dbl ****)array_alloc(4, exo->num_elems, ip_total, DIM, DIM, sizeof(dbl));
      goma_ctx->fem.evpl_glob[mn]->dTT_dx_glob =
          (dbl ******)array_alloc(6, exo->num_elems, ip_total, DIM, DIM, DIM, MDE, sizeof(dbl));
      goma_ctx->fem.evpl_glob[mn]->dTT_dx_old_glob =
          (dbl ******)array_alloc(6, exo->num_elems, ip_total, DIM, DIM, DIM, MDE, sizeof(dbl));
      goma_ctx->fem.evpl_glob[mn]->dTT_dc_glob = (dbl ******)array_alloc(
          6, exo->num_elems, ip_total, DIM, DIM, MAX_CONC, MDE, sizeof(dbl));
      goma_ctx->fem.evpl_glob[mn]->dTT_dc_old_glob = (dbl ******)array_alloc(
          6, exo->num_elems, ip_total, DIM, DIM, MAX_CONC, MDE, sizeof(dbl));

      for (i = 0; i < exo->num_elems; i++) {
        for (ip = 0; ip < ip_total; ip++) {
          for (a = 0; a < DIM; a++) {
            for (b = 0; b < DIM; b++) {
              goma_ctx->fem.evpl_glob[mn]->F_vp_old_glob[i][ip][a][b] = 0.;
              goma_ctx->fem.evpl_glob[mn]->F_vp_glob[i][ip][a][b] = 0.;
              goma_ctx->fem.evpl_glob[mn]->TT_glob[i][ip][a][b] = 0.;
              goma_ctx->fem.evpl_glob[mn]->TT_old_glob[i][ip][a][b] = 0.;

              for (k = 0; k < MDE; k++) {
                for (c = 0; c < DIM; c++) {
                  goma_ctx->fem.evpl_glob[mn]->dTT_dx_glob[i][ip][a][b][c][k] = 0.;
                  goma_ctx->fem.evpl_glob[mn]->dTT_dx_old_glob[i][ip][a][b][c][k] = 0.;
                }
                for (w = 0; w < MAX_CONC; w++) {
                  goma_ctx->fem.evpl_glob[mn]->dTT_dc_glob[i][ip][a][b][w][k] = 0.;
                  goma_ctx->fem.evpl_glob[mn]->dTT_dc_old_glob[i][ip][a][b][w][k] = 0.;
                }
              }
            }
          }
        }
      }
    }
  }

  return (status);
}
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/

int elc_rs_alloc(void) {
  int sz;
  int mn;
  int status;
  static char yo[] = "elc_alloc";

  status = 0;

  /*
   * Elastic_Constitutive____________________________________________________
   */

  sz = sizeof(struct Elastic_Constitutive *);
  goma_ctx->fem.elc_rs_glob = (struct Elastic_Constitutive **)smalloc(MAX_NUMBER_MATLS * sz);

  sz = sizeof(struct Elastic_Constitutive);
  for (mn = 0; mn < MAX_NUMBER_MATLS; mn++) {
    goma_ctx->fem.elc_rs_glob[mn] = (struct Elastic_Constitutive *)smalloc(sz);
    init_Elastic_Constitutive(goma_ctx->fem.elc_rs_glob[mn]);
  }

  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Elastic Constitutive RS @ %p has %d bytes", yo,
            (void *)goma_ctx->fem.gn_glob, sz);
  }

  if (goma_ctx->fem.elc_rs_glob == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Elastic_Constitutive");
  }

  return (status);
}

int tran_alloc(void) {
  int sz;
  int status;
  static char yo[] = "tran_alloc";

  status = 0;

  /*
   * Transient_Information______________________________________________________
   */

  sz = sizeof(struct Transient_Information);
  goma_ctx->fem.tran = (struct Transient_Information *)array_alloc(1, 1, sz);
  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Transient_Information @ %p has %d bytes", yo, (void *)goma_ctx->fem.tran,
            sz);
  }

  if (goma_ctx->fem.tran == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Transient_Information");
  }

  /*
   * Element_Quality_Metrics___________________________________________________
   */
  sz = sizeof(struct Element_Quality_Metrics);
  goma_ctx->fem.eqm = alloc_struct_1(struct Element_Quality_Metrics, 1);

  if (goma_ctx->debug_flag) {
    P0PRINTF("%s: Element_Quality_Metrics @ %p has %d bytes", yo, (void *)goma_ctx->fem.eqm, sz);
  }

  if (goma_ctx->fem.eqm == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Element_Quality_Metrics");
  }

  return (status);
}
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/

int libio_alloc(void) {
  int sz;
  int status;
  static char yo[] = "libio_alloc";

  status = 0;

  /*
   * Library_IO
   */

  sz = sizeof(struct Library_IO);
  goma_ctx->fem.libio = (struct Library_IO *)array_alloc(1, 1, sz);
  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Library_IO @ %p has %d bytes", yo, (void *)goma_ctx->fem.libio, sz);
  }

  if (goma_ctx->fem.libio == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Library_IO");
  }

  return (status);
}
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/

int eigen_alloc(void) {
  int sz;
  int status;
  static char yo[] = "eigen_alloc";

  status = 0;

  /*
   * Eigensolver_Info
   */

  sz = sizeof(struct Eigensolver_Info);
  goma_ctx->fem.eigen = (struct Eigensolver_Info *)array_alloc(1, 1, sz);
  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Eigensolver_Info @ %p has %d bytes", yo, (void *)goma_ctx->fem.eigen, sz);
  }

  if (goma_ctx->fem.eigen == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Eigensolver_Info");
  }

  return (status);
}
/*************************************************************************/
/*************************************************************************/
/*************************************************************************/

int cont_alloc(void) {
  int sz;
  int status;
  static char yo[] = "cont_alloc";

  status = 0;

  /*
   * Continuation_Information
   */

  sz = sizeof(struct Continuation_Information);
  goma_ctx->fem.cont = (struct Continuation_Information *)array_alloc(1, 1, sz);
  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Continuation_Information @ %p has %d bytes", yo,
            (void *)goma_ctx->fem.cont, sz);
  }

  if (goma_ctx->fem.cont == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Continuation_Information");
  }

  return (status);
}
/*************************************************************************/
/*************************************************************************/
/*************************************************************************/

int loca_alloc(void) {
  int sz;
  int status;
  static char yo[] = "loca_alloc";

  status = 0;

  /*
   * Loca_Input
   */

  sz = sizeof(struct Loca_Input);
  goma_ctx->fem.loca_in = (struct Loca_Input *)array_alloc(1, 1, sz);
  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Loca_Input @ %p has %d bytes", yo, (void *)goma_ctx->fem.cont, sz);
  }

  if (goma_ctx->fem.loca_in == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Loca_Input");
  }

  return (status);
}
/***************************************************************************/
/***************************************************************************/
/***************************************************************************/
static void Element_Indices_alloc(struct Element_Indices *ei_ptr) {

  ei_ptr->Num_Lvdesc_Per_Var_Type = alloc_int_1(MAX_VARIABLE_TYPES + MAX_CONC, 0);
  ei_ptr->Lvdesc_First_Var_Type = alloc_int_1(MAX_VARIABLE_TYPES + MAX_CONC, -1);
  ei_ptr->Lvdesc_to_ledof = alloc_int_2(MAX_LOCAL_VAR_DESC, MDE, -1);
  ei_ptr->Lvdesc_to_lvdof = alloc_int_2(MAX_LOCAL_VAR_DESC, MDE, -1);
  ei_ptr->Lvdesc_to_Var_Type = alloc_int_1(MAX_LOCAL_VAR_DESC, -1);
  ei_ptr->Lvdesc_to_MatID = alloc_int_1(MAX_LOCAL_VAR_DESC, -1);
  ei_ptr->Lvdesc_Numdof = alloc_int_1(MAX_LOCAL_VAR_DESC, 0);
  ei_ptr->Lvdesc_to_Gnn = alloc_int_2(MAX_LOCAL_VAR_DESC, MDE, -1);
  ei_ptr->Lvdesc_to_Gun = alloc_int_2(MAX_LOCAL_VAR_DESC, MDE, -1);
  ei_ptr->Lvdesc_to_MFSubvar = alloc_int_1(MAX_LOCAL_VAR_DESC, 0);
  ei_ptr->Lvdesc_to_Lnn = alloc_int_2(MAX_LOCAL_VAR_DESC, MDE, -1);
  ei_ptr->Lvdesc_Lnn_to_Offset = alloc_int_2(MAX_LOCAL_VAR_DESC, MDE, -1);
  ei_ptr->Lvdesc_Lnn_to_lvdof = alloc_int_2(MAX_LOCAL_VAR_DESC, MDE, -1);
  ei_ptr->lvdof_to_lvdesc = alloc_int_2(MAX_VARIABLE_TYPES + MAX_CONC, MDE, -1);
  ei_ptr->lvdof_to_lvdesc_dof = alloc_int_2(MAX_VARIABLE_TYPES + MAX_CONC, MDE, -1);
  ei_ptr->Lvdesc_Lnn_Numdof = alloc_int_2(MAX_LOCAL_VAR_DESC, MDE, -1);
  ei_ptr->Lvdesc_vd_ptr = (VARIABLE_DESCRIPTION_STRUCT **)alloc_ptr_1(MAX_LOCAL_VAR_DESC);
  /*
   * At this point we don't know the value of Num_Var_Info_Records
   * so we can't malloc the following:
   *       ei_ptr->VDindex_to_Lvdesc = alloc_int_1(Num_Var_Info_Records, -1);
   */

  ei_ptr->owningElementForColVar = alloc_int_1(MAX_VARIABLE_TYPES + MAX_CONC, 0);
  /*
   * For each equation/variable, tell how many dofs must be accounted for
   * in this element and their names(nodes)...
   *
   */
  ei_ptr->dof = alloc_int_1(MAX_VARIABLE_TYPES, 0);
  ei_ptr->dof_ext = alloc_int_1(MAX_EXTERNAL_FIELD, 0);

  ei_ptr->Baby_Dolphin = alloc_int_2(MAX_VARIABLE_TYPES, MDE, -1);

  /*
   * In more detail, tell the list of nodes associated with those dofs
   * for each variable.
   */
  ei_ptr->dof_list = alloc_int_2(MAX_VARIABLE_TYPES, MDE, -1);

  /*
   * Finally, pointers into the global unknown arrays for each dof in this
   * element...
   */
  ei_ptr->gnn_list = alloc_int_2(MAX_VARIABLE_TYPES, MDE, -1);

  /*
   * Finally, pointers into the global unknown arrays for each dof in this
   * element...
   */
  ei_ptr->gun_list = alloc_int_2(MAX_VARIABLE_TYPES, MDE, -1);
  ei_ptr->ln_to_dof = alloc_int_2(MAX_VARIABLE_TYPES, MDE, -1);
  ei_ptr->ln_to_first_dof = alloc_int_2(MAX_VARIABLE_TYPES, MDE, -1);
  ei_ptr->lvdof_to_row_lvdof = alloc_int_2(MAX_VARIABLE_TYPES, MDE, -1);
  ei_ptr->lvdof_to_ledof = alloc_int_2(MAX_VARIABLE_TYPES, MDE, -1);

  ei_ptr->owned_ledof = alloc_int_1(MDE * MAX_PROB_VAR, 1);
  ei_ptr->ieqn_ledof = alloc_int_1(MDE * MAX_PROB_VAR, -1);
  ei_ptr->matID_ledof = alloc_int_1(MDE * MAX_PROB_VAR, -1);
  ei_ptr->active_interp_ledof = alloc_int_1(MDE * MAX_PROB_VAR, -1);

  ei_ptr->MFsubvar_Offset = alloc_int_2(MAX_CONC, MDE, 0);
}

/***************************************************************************/
/***************************************************************************/
/***************************************************************************/

int assembly_alloc(Exo_DB *exo)

/********************************************************************
 *
 * assembly_alloc:
 *
 *
 ********************************************************************/
{
  int vi, sz;
  int type; /* index for unique basis functions */
  int status = 0;
  int interp; /* index for interpolation order */
  int si;     /* index for element shape */
  int mn;
  int ipore;
  int num_species_eqn; /* active number of species eqn */
  int imtrx;

  /*
   * The problem description has already been set up. But we need to access
   * it here...
   */

  static char yo[] = "assembly_alloc";

  /*
   * These are critical for the element dof pointers...
   */
  goma_ctx->fem.p0 = &zero;

  if (goma_ctx->debug_flag) {
    DPRINTF(stderr, "%s...\n", yo);
  }

  /*
   * Element_Indices___________________________________________________________
   */

  goma_ctx->fem.ei =
      malloc(sizeof(struct Element_Indices *) * goma_ctx->fem.upd->Total_Num_Matrices);
  for (goma_ctx->fem.pg->imtrx = 0; goma_ctx->fem.pg->imtrx < goma_ctx->fem.upd->Total_Num_Matrices;
       goma_ctx->fem.pg->imtrx++) {
    goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx] = alloc_struct_1(struct Element_Indices, 1);
    Element_Indices_alloc(goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]);
  }
  goma_ctx->fem.pg->imtrx = 0;

  goma_ctx->fem.ei_related = (struct Element_Indices **)alloc_ptr_1(MAX_ELEMENT_INDICES_RELATED);
  for (mn = 0; mn < MAX_ELEMENT_INDICES_RELATED; mn++) {
    goma_ctx->fem.ei_related[mn] = alloc_struct_1(struct Element_Indices, 1);
    Element_Indices_alloc(goma_ctx->fem.ei_related[mn]);
  }

  /*
   * Element_Variable_Pointers________________________________________________
   */

  sz = sizeof(struct Element_Variable_Pointers);
  goma_ctx->fem.esp_old = (struct Element_Variable_Pointers *)array_alloc(1, 1, sz);
  goma_ctx->fem.esp_dot = (struct Element_Variable_Pointers *)array_alloc(1, 1, sz);
  goma_ctx->fem.esp_dbl_dot = (struct Element_Variable_Pointers *)array_alloc(1, 1, sz);
  goma_ctx->fem.evp = (struct Element_Variable_Pointers *)array_alloc(1, 1, sz);

  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Element_Variable_Pointers @ %p has %d bytes", yo,
            (void *)goma_ctx->fem.esp_old, sz);
  }

  if ((goma_ctx->fem.esp_old == NULL) || (goma_ctx->fem.esp_dot == NULL)) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Element_Variable_Pointers");
  }

  /*
   * Element_Stiffness_Pointers________________________________________________
   */
  sz = sizeof(struct Element_Stiffness_Pointers);
  goma_ctx->fem.esp = alloc_struct_1(struct Element_Stiffness_Pointers, 1);

  if (goma_ctx->debug_flag) {
    P0PRINTF("%s: Element_Stiffness_Pointers @ %p has %d bytes", yo, (void *)goma_ctx->fem.esp, sz);
  }

  /*
   *  num_species_eqn is equal to the maximum number of species equations
   *  in any one domain
   */
  num_species_eqn = goma_ctx->fem.upd->Max_Num_Species_Eqn;

  sz = MDE;

  /* MMH I have included some coupling between particle velocity and
   * other things.  I have skipped over a bunch (like energy) b/c there
   * are no models, yet, that include both energy/temperature and
   * particle velocity
   */

  for (imtrx = 0; imtrx < goma_ctx->fem.upd->Total_Num_Matrices; imtrx++) {
    /* ENERGY */
    if (goma_ctx->problem.num_var_in_type[imtrx][TEMPERATURE]) {
      goma_ctx->fem.esp->T = (dbl **)alloc_ptr_1(MDE);
    }

    /* MOMENTUM  */
    if (goma_ctx->problem.num_var_in_type[imtrx][VELOCITY1]) {
      goma_ctx->fem.esp->v = (dbl ***)alloc_ptr_2(goma_ctx->problem.vim, MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][USTAR]) {
      goma_ctx->fem.esp->v_star = (dbl ***)alloc_ptr_2(goma_ctx->problem.vim, MDE);
    }

    /* MMH
     * Some of these are just leftovers from copying the velocity stuff.  I left
     * them here in case their particle equivalents are incorporated later.
     */
    /* PARTICLE MOMENTUM */
    if (goma_ctx->problem.num_var_in_type[imtrx][PVELOCITY1]) {
      goma_ctx->fem.esp->pv = (dbl ***)alloc_ptr_2(goma_ctx->problem.vim, MDE);
    }

    /* MESH_DISPLACEMENT  */
    if (goma_ctx->problem.num_var_in_type[imtrx][MESH_DISPLACEMENT1]) {
      goma_ctx->fem.esp->d = (dbl ***)alloc_ptr_2(goma_ctx->problem.vim, MDE);
    }

    /* SOLID_DISPLACEMENT  */
    if (goma_ctx->problem.num_var_in_type[imtrx][SOLID_DISPLACEMENT1]) {
      goma_ctx->fem.esp->d_rs = (dbl ***)alloc_ptr_2(goma_ctx->problem.vim, MDE);
    }

    /* SPECIES CONTINUITY */
    if (goma_ctx->problem.num_var_in_type[imtrx][MASS_FRACTION]) {
      goma_ctx->fem.esp->c = (dbl ***)alloc_ptr_2(num_species_eqn, MDE);
    }

    /*CONTINUITY */
    if (goma_ctx->problem.num_var_in_type[imtrx][PRESSURE]) {
      goma_ctx->fem.esp->P = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][PSTAR]) {
      goma_ctx->fem.esp->P_star = (dbl **)alloc_ptr_1(MDE);
    }

    /* POLYMER STRESS for all modes */
    if (goma_ctx->problem.num_var_in_type[imtrx][POLYMER_STRESS11]) {
      int sdim = goma_ctx->problem.vim;
      for (int i = 0; i < goma_ctx->fem.upd->Total_Num_Matrices; i++) {
        if (goma_ctx->problem.num_var_in_type[i][FILM_HEIGHT])
          sdim = 3;
      }
      goma_ctx->fem.esp->S = (dbl *****)array_alloc(4, MAX_MODES, sdim, sdim, MDE, sizeof(dbl *));
      (void)memset(goma_ctx->fem.esp->S[0][0][0], 0, MAX_MODES * sdim * sdim * MDE * sizeof(dbl *));
    }

    /* VELOCITY_GRADIENT */
    if (goma_ctx->problem.num_var_in_type[imtrx][VELOCITY_GRADIENT11]) {
      int sdim = goma_ctx->problem.vim;
      for (int i = 0; i < goma_ctx->fem.upd->Total_Num_Matrices; i++) {
        if (goma_ctx->problem.num_var_in_type[i][FILM_HEIGHT])
          sdim = 3;
      }
      goma_ctx->fem.esp->G = (dbl ****)array_alloc(3, sdim, sdim, MDE, sizeof(dbl *));
      (void)memset(goma_ctx->fem.esp->G[0][0], 0, sdim * sdim * MDE * sizeof(dbl *));
    }

    /* POTENTIAL */
    if (goma_ctx->problem.num_var_in_type[imtrx][VOLTAGE]) {
      goma_ctx->fem.esp->V = (dbl **)alloc_ptr_1(MDE);
    }

    /* SURF_CHARGE */
    if (goma_ctx->problem.num_var_in_type[imtrx][SURF_CHARGE]) {
      goma_ctx->fem.esp->qs = (dbl **)alloc_ptr_1(MDE);
    }

    /* FILL */
    if (goma_ctx->problem.num_var_in_type[imtrx][FILL]) {
      goma_ctx->fem.esp->F = (dbl **)alloc_ptr_1(MDE);
    }

    /* SHEAR_RATE INVARIANT */
    if (goma_ctx->problem.num_var_in_type[imtrx][SHEAR_RATE]) {
      goma_ctx->fem.esp->SH = (dbl **)alloc_ptr_1(MDE);
    }

    /* ENORM, |E| */
    if (goma_ctx->problem.num_var_in_type[imtrx][ENORM]) {
      goma_ctx->fem.esp->Enorm = (dbl **)alloc_ptr_1(MDE);
    }

    /* LEVEL SET CURVATURE */
    if (goma_ctx->problem.num_var_in_type[imtrx][CURVATURE]) {
      goma_ctx->fem.esp->H = (dbl **)alloc_ptr_1(MDE);
    }

    /* LEVEL SET NORMAL VECTOR */
    if (goma_ctx->problem.num_var_in_type[imtrx][NORMAL1]) {
      goma_ctx->fem.esp->n = (dbl ***)alloc_ptr_2(goma_ctx->problem.vim, MDE);
    }

    /* POROUS MEDIA VARS */
    if (goma_ctx->problem.num_var_in_type[imtrx][POR_LIQ_PRES]) {
      goma_ctx->fem.esp->p_liq = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][POR_GAS_PRES]) {
      goma_ctx->fem.esp->p_gas = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][POR_POROSITY]) {
      goma_ctx->fem.esp->porosity = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][POR_TEMP]) {
      goma_ctx->fem.esp->T = (dbl **)alloc_ptr_1(MDE);
    }

    /* VORTICITY PRINCIPLE FLOW DIRECTION
     * This is always 3D */
    if (goma_ctx->problem.num_var_in_type[imtrx][VORT_DIR1]) {
      goma_ctx->fem.esp->vd = (dbl ***)alloc_ptr_2(DIM, MDE);
    }

    /* Lagrange Multipliers */
    if (goma_ctx->problem.num_var_in_type[imtrx][LAGR_MULT1]) {
      goma_ctx->fem.esp->lm = (dbl ***)alloc_ptr_2(goma_ctx->problem.vim, MDE);
    }

    /* Structural Shell equations */
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_CURVATURE]) {
      goma_ctx->fem.esp->sh_K = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_CURVATURE2]) {
      goma_ctx->fem.esp->sh_K2 = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_TENSION]) {
      goma_ctx->fem.esp->sh_tens = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_X]) {
      goma_ctx->fem.esp->sh_x = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_Y]) {
      goma_ctx->fem.esp->sh_y = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_USER]) {
      goma_ctx->fem.esp->sh_u = (dbl **)alloc_ptr_1(MDE);
    }

    /* Shell orientation angles */
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_ANGLE1]) {
      goma_ctx->fem.esp->sh_ang = (dbl ***)alloc_ptr_2(DIM - 1, MDE);
    }

    /*Sundry pieces for Surface Rheological Const. Eqn. */
    if (goma_ctx->problem.num_var_in_type[imtrx][R_SHELL_SURF_DIV_V]) {
      goma_ctx->fem.esp->div_s_v = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][R_SHELL_SURF_CURV]) {
      goma_ctx->fem.esp->curv = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][R_N_DOT_CURL_V]) {
      goma_ctx->fem.esp->n_dot_curl_s_v = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][R_GRAD_S_V_DOT_N1]) {
      goma_ctx->fem.esp->grad_v_dot_n = (dbl ***)alloc_ptr_2(DIM, MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][R_SHELL_DIFF_FLUX]) {
      goma_ctx->fem.esp->sh_J = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][R_SHELL_DIFF_CURVATURE]) {
      goma_ctx->fem.esp->sh_Kd = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][R_SHELL_NORMAL1]) {
      goma_ctx->fem.esp->n = (dbl ***)alloc_ptr_2(DIM, MDE);
    }

    /* EIGENVALUE FOR VORTICITY PRINCIPLE FLOW DIRECTION
     * This is always 3D */
    if (goma_ctx->problem.num_var_in_type[imtrx][VORT_LAMBDA]) {
      goma_ctx->fem.esp->vlambda = (dbl **)alloc_ptr_1(MDE);
    }

    /* BOND Evolution
     * associated with structure formation during flow */
    if (goma_ctx->problem.num_var_in_type[imtrx][BOND_EVOLUTION]) {
      goma_ctx->fem.esp->nn = (dbl **)alloc_ptr_1(MDE);
    }

    /* Extension Velocity */
    if (goma_ctx->problem.num_var_in_type[imtrx][EXT_VELOCITY]) {
      goma_ctx->fem.esp->ext_v = (dbl **)alloc_ptr_1(MDE);
    }

    /* Electric Field */
    if (goma_ctx->problem.num_var_in_type[imtrx][EFIELD1]) {
      goma_ctx->fem.esp->E_field = (dbl ***)alloc_ptr_2(goma_ctx->problem.vim, MDE);
    }

    /* Phase function */
    if (goma_ctx->problem.num_var_in_type[imtrx][PHASE1]) {
      goma_ctx->fem.esp->pF = (dbl ***)alloc_ptr_2(goma_ctx->fem.pfd->num_phase_funcs, MDE);
    }
    /* Acoustic pressure */
    if (goma_ctx->problem.num_var_in_type[imtrx][ACOUS_PREAL]) {
      goma_ctx->fem.esp->apr = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][ACOUS_PIMAG]) {
      goma_ctx->fem.esp->api = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][ACOUS_REYN_STRESS]) {
      goma_ctx->fem.esp->ars = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][EM_CONT_REAL]) {
      goma_ctx->fem.esp->epr = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][EM_CONT_IMAG]) {
      goma_ctx->fem.esp->epi = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_BDYVELO]) {
      goma_ctx->fem.esp->sh_bv = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_LUBP]) {
      goma_ctx->fem.esp->sh_p = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][LUBP]) {
      goma_ctx->fem.esp->lubp = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][LUBP_2]) {
      goma_ctx->fem.esp->lubp_2 = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_FILMP]) {
      goma_ctx->fem.esp->sh_fp = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_FILMH]) {
      goma_ctx->fem.esp->sh_fh = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_PARTC]) {
      goma_ctx->fem.esp->sh_pc = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_SAT_CLOSED]) {
      goma_ctx->fem.esp->sh_sat_closed = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_PRESS_OPEN]) {
      goma_ctx->fem.esp->sh_p_open = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_PRESS_OPEN_2]) {
      goma_ctx->fem.esp->sh_p_open_2 = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_TEMPERATURE]) {
      goma_ctx->fem.esp->sh_t = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_DELTAH]) {
      goma_ctx->fem.esp->sh_dh = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_LUB_CURV]) {
      goma_ctx->fem.esp->sh_l_curv = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_LUB_CURV_2]) {
      goma_ctx->fem.esp->sh_l_curv_2 = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_SAT_GASN]) {
      goma_ctx->fem.esp->sh_sat_gasn = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][POR_SINK_MASS]) {
      goma_ctx->fem.esp->sink_mass = (dbl **)alloc_ptr_1(MDE);
    }
    /* Poynting Vector  */
    if (goma_ctx->problem.num_var_in_type[imtrx][LIGHT_INTP] ||
        goma_ctx->problem.num_var_in_type[imtrx][LIGHT_INTM] ||
        goma_ctx->problem.num_var_in_type[imtrx][LIGHT_INTD]) {
      goma_ctx->fem.esp->poynt = (dbl ***)alloc_ptr_2(goma_ctx->problem.vim, MDE);
    }
    /* EM_wave components  */
    if (goma_ctx->problem.num_var_in_type[imtrx][EM_E1_REAL] ||
        goma_ctx->problem.num_var_in_type[imtrx][EM_E2_REAL] ||
        goma_ctx->problem.num_var_in_type[imtrx][EM_E3_REAL]) {
      goma_ctx->fem.esp->em_er = (dbl ***)alloc_ptr_2(goma_ctx->problem.vim, MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][EM_E1_IMAG] ||
        goma_ctx->problem.num_var_in_type[imtrx][EM_E2_IMAG] ||
        goma_ctx->problem.num_var_in_type[imtrx][EM_E3_IMAG]) {
      goma_ctx->fem.esp->em_ei = (dbl ***)alloc_ptr_2(goma_ctx->problem.vim, MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][EM_H1_REAL] ||
        goma_ctx->problem.num_var_in_type[imtrx][EM_H2_REAL] ||
        goma_ctx->problem.num_var_in_type[imtrx][EM_H3_REAL]) {
      goma_ctx->fem.esp->em_hr = (dbl ***)alloc_ptr_2(goma_ctx->problem.vim, MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][EM_H1_IMAG] ||
        goma_ctx->problem.num_var_in_type[imtrx][EM_H2_IMAG] ||
        goma_ctx->problem.num_var_in_type[imtrx][EM_H3_IMAG]) {
      goma_ctx->fem.esp->em_hi = (dbl ***)alloc_ptr_2(goma_ctx->problem.vim, MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][TFMP_PRES]) {
      goma_ctx->fem.esp->tfmp_pres = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][TFMP_SAT]) {
      goma_ctx->fem.esp->tfmp_sat = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][RESTIME]) {
      goma_ctx->fem.esp->restime = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][MOMENT0]) {
      goma_ctx->fem.esp->moment = (dbl ***)alloc_ptr_2(MAX_MOMENTS, MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][DENSITY_EQN]) {
      goma_ctx->fem.esp->rho = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_SHEAR_TOP]) {
      goma_ctx->fem.esp->sh_shear_top = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_SHEAR_BOT]) {
      goma_ctx->fem.esp->sh_shear_bot = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_CROSS_SHEAR]) {
      goma_ctx->fem.esp->sh_cross_shear = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][MAX_STRAIN]) {
      goma_ctx->fem.esp->max_strain = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][CUR_STRAIN]) {
      goma_ctx->fem.esp->cur_strain = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_SAT_1]) {
      goma_ctx->fem.esp->sh_sat_1 = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_SAT_2]) {
      goma_ctx->fem.esp->sh_sat_2 = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][SHELL_SAT_3]) {
      goma_ctx->fem.esp->sh_sat_3 = (dbl **)alloc_ptr_1(MDE);
    }

    if (goma_ctx->problem.num_var_in_type[imtrx][EDDY_NU]) {
      goma_ctx->fem.esp->eddy_nu = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][TURB_K]) {
      goma_ctx->fem.esp->turb_k = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][TURB_OMEGA]) {
      goma_ctx->fem.esp->turb_omega = (dbl **)alloc_ptr_1(MDE);
    }
    if (goma_ctx->problem.num_var_in_type[imtrx][FILM_HEIGHT]) {
      goma_ctx->fem.esp->film_height = (dbl **)alloc_ptr_1(MDE);
    }

  } /* End of loop over matrices */

  /*
   * Action_Flags______________________________________________________________
   */
  goma_ctx->fem.af = alloc_struct_1(struct Action_Flags, 1);
  if (goma_ctx->debug_flag) {
    P0PRINTF("%s: Action_Flags @ %p has %lu bytes", yo, (void *)goma_ctx->fem.af,
             (long unsigned int)sizeof(struct Action_Flags));
  }

  /*
   * Basis functions: allocate space only for a list of unique basis functions
   * required to assemble terms for this problem. Then, depending on the
   * particular weighing and interpolation to be used for each variable,
   * point to the appropriate member of the unique set.
   *
   * Here, the "bfd" point to real stuff.
   *
   * Then, the "bf" just point to the right place (see bf_init)...
   */

  /*
   * Hardwired for now...later, load these up based on the EXODUS II database
   */

  /* This was the old way:
    if ( Num_Dim == 2 )
      {
        Num_Shapes       = 1;
        Unique_Shapes[0] = QUADRILATERAL;
        **      Unique_Shapes[1] = TRIANGLE; **
      }
    else if ( Num_Dim == 3 )
      {
        Num_Shapes       = 1;
        Unique_Shapes[0] = HEXAHEDRON;
        **      Unique_Shapes[1] = TETRAHEDRON; **
        **      Unique_Shapes[2] = PRISM; **
      }
    else if ( Num_Dim == 1 )
      {
        Num_Shapes       = 1;
        Unique_Shapes[0] = LINE_SEGMENT;
      }
    else
      {
        GOMA_EH(GOMA_ERROR, "Cannot classify shapes...");
      }
  */

  /* Now, here is the new way: */
  goma_ctx->problem.num_shapes = shape_list(exo);

  /*
   * To determine how many types of basis functions we'll need to represent,
   * multiply the number of basic element shapes by the number of different
   * interpolations that have been specified...
   */
  goma_ctx->problem.num_basis_functions =
      goma_ctx->problem.num_shapes * goma_ctx->problem.num_interpolations;

  goma_ctx->fem.bfd = (struct Basis_Functions **)alloc_ptr_1(goma_ctx->problem.num_basis_functions);
  goma_ctx->fem.bfi = (struct Basis_Functions **)alloc_ptr_1(MAX_INTERP_TYPES);
  goma_ctx->fem.bf = (struct Basis_Functions **)alloc_ptr_1(MAX_VARIABLE_TYPES);
  goma_ctx->fem.bfex = (struct Basis_Functions **)alloc_ptr_1(MAX_EXTERNAL_FIELD);

  sz = sizeof(struct Basis_Functions);

  /*
   * Now allocate and do some initialization of the fundamental basis functions
   */

  type = 0;
  for (si = 0; si < goma_ctx->problem.num_shapes; si++) {
    for (interp = 0; interp < goma_ctx->problem.num_interpolations; interp++) {
      goma_ctx->fem.bfd[type] = alloc_struct_1(struct Basis_Functions, 1);
      goma_ctx->fem.bfd[type]->Var_Type_MatID = alloc_int_1(goma_ctx->fem.upd->Num_Mat, -1);
      goma_ctx->fem.bfd[type]->interpolation = goma_ctx->problem.unique_interpolations[interp];
      goma_ctx->fem.bfd[type]->element_shape = goma_ctx->problem.unique_shapes[si];
      goma_ctx->fem.bfd[type]->Max_Dofs_Interpolation = getdofs(
          goma_ctx->problem.unique_shapes[si], goma_ctx->problem.unique_interpolations[interp]);
      /*
       * Find a representative variable type in each material
       * that employs the current interpolation.
       * Store it in the structure.
       *
       * This is an overly broad interpretation. Specifically,
       * the intersection of an element shape and a material
       * type is never tried.
       */
      for (mn = 0; mn < goma_ctx->fem.upd->Num_Mat; mn++) {
        for (imtrx = 0; imtrx < goma_ctx->fem.upd->Total_Num_Matrices; imtrx++) {
          for (vi = 0;
               ((goma_ctx->fem.bfd[type]->Var_Type_MatID[mn] == -1) && (vi < MAX_VARIABLE_TYPES));
               vi++) {
            if (goma_ctx->fem.pd_glob[mn]->i[imtrx][vi] == goma_ctx->fem.bfd[type]->interpolation) {
              goma_ctx->fem.bfd[type]->Var_Type_MatID[mn] = vi;
            }
          }
        }
      }
      type++;
    }
  }

  /* set pointers for basis function interpolation types */
  for (interp = 0; interp < goma_ctx->problem.num_interpolations; interp++) {
    type = goma_ctx->problem.unique_interpolations[interp];
    goma_ctx->fem.bfi[type] = goma_ctx->fem.bfd[interp];
  }
  /* so now bfi[pd_glob[mn]->i[eqn]] points to a basis function with
     the correct interpolation */

  /*
   * Field_Variables___________________________________________________________
   */

  goma_ctx->fem.fv = alloc_struct_1(struct Field_Variables, 1);
  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Field_Variables @ %p has %ld bytes", yo, (void *)goma_ctx->fem.fv,
            (long int)sizeof(struct Field_Variables));
  }
  if (goma_ctx->fem.fv == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Field_Variables");
  }
  memset(goma_ctx->fem.fv, 0, sizeof(struct Field_Variables));

  goma_ctx->fem.fv_sens = alloc_struct_1(struct Field_Variables, 1);
  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Field_Variables Sensitivity @ %p has %lu bytes", yo,
            (void *)goma_ctx->fem.fv_sens, sizeof(struct Field_Variables));
  }
  if (goma_ctx->fem.fv_sens == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Field_Variables (sens)");
  }
  memset(goma_ctx->fem.fv_sens, 0, sizeof(struct Field_Variables));

  sz = sizeof(struct Diet_Field_Variables);
  goma_ctx->fem.fv_old = alloc_struct_1(struct Diet_Field_Variables, 1);
  goma_ctx->fem.fv_dot = alloc_struct_1(struct Diet_Field_Variables, 1);
  goma_ctx->fem.fv_dot_dot = alloc_struct_1(struct Diet_Field_Variables, 1);
  goma_ctx->fem.fv_dot_old = alloc_struct_1(struct Diet_Field_Variables, 1);
  goma_ctx->fem.fv_dot_dot_old = alloc_struct_1(struct Diet_Field_Variables, 1);
  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Diet_Field_Variables @ %p has %d bytes", yo, (void *)goma_ctx->fem.fv_old,
            sz);
  }
  if (goma_ctx->fem.fv_old == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Diet_Field_Variables");
  }

  /*
   * Porous_Media_Variables__(if needed)_________________________________________
   */
  for (mn = 0; mn < goma_ctx->fem.upd->Num_Mat && goma_ctx->fem.pmv == NULL; mn++) {
    if (goma_ctx->fem.mp_glob[mn]->PorousMediaType == POROUS_SATURATED ||
        goma_ctx->fem.mp_glob[mn]->PorousMediaType == POROUS_UNSATURATED ||
        goma_ctx->fem.mp_glob[mn]->PorousMediaType == POROUS_TWO_PHASE ||
        goma_ctx->fem.mp_glob[mn]->PorousMediaType == POROUS_SHELL_UNSATURATED) {
      goma_ctx->fem.pmv = alloc_struct_1(struct Porous_Media_Variables, 1);
      goma_ctx->fem.pmv_old = alloc_struct_1(struct Porous_Media_Variables, 1);
      break;
    }
  }
  for (mn = 0; mn < goma_ctx->fem.upd->Num_Mat && goma_ctx->fem.pmv_ml == NULL; mn++) {
    if (goma_ctx->fem.mp_glob[mn]->PorousMediaType == POROUS_SATURATED ||
        goma_ctx->fem.mp_glob[mn]->PorousMediaType == POROUS_UNSATURATED ||
        goma_ctx->fem.mp_glob[mn]->PorousMediaType == POROUS_TWO_PHASE ||
        goma_ctx->fem.mp_glob[mn]->PorousMediaType == POROUS_SHELL_UNSATURATED) {
      if ((goma_ctx->fem.mp_glob[mn]->Porous_Mass_Lump) ||
          (goma_ctx->fem.mp_glob[mn]->PorousMediaType == POROUS_SHELL_UNSATURATED)) {
        goma_ctx->fem.pmv_ml = alloc_struct_1(PMV_ML_STRUCT, 1);
      }
      break;
    }
  }
  for (mn = 0; mn < goma_ctx->fem.upd->Num_Mat && goma_ctx->fem.pmv_hyst == NULL; mn++) {
    if (goma_ctx->fem.mp_glob[mn]->PorousMediaType == POROUS_SHELL_UNSATURATED) {
      for (ipore = 0; ipore < goma_ctx->fem.pd_glob[mn]->Num_Porous_Shell_Eqn; ipore++) {
        if ((goma_ctx->fem.mp_glob[mn]->PorousShellCapPresModel[ipore] == VAN_GENUCHTEN_HYST) ||
            (goma_ctx->fem.mp_glob[mn]->PorousShellCapPresModel[ipore] == VAN_GENUCHTEN_HYST_EXT)) {
          goma_ctx->fem.pmv_hyst = alloc_struct_1(struct Porous_Media_Variables_Hysteresis, 1);
          break;
        }
      }
      if (goma_ctx->fem.pmv_hyst != NULL) {
        for (ipore = 0; ipore < goma_ctx->fem.pd_glob[mn]->Num_Porous_Shell_Eqn; ipore++) {
          goma_ctx->fem.pmv_hyst->curve_type[ipore] = alloc_int_1(exo->num_nodes, -1);
          goma_ctx->fem.pmv_hyst->curve_type_old[ipore] = alloc_int_1(exo->num_nodes, -1);
          goma_ctx->fem.pmv_hyst->curve_switch[ipore] = alloc_int_1(exo->num_nodes, -1);
          goma_ctx->fem.pmv_hyst->num_switch[ipore] = alloc_int_1(exo->num_nodes, -1);

          goma_ctx->fem.pmv_hyst->sat_switch[ipore] = alloc_dbl_1(exo->num_nodes, 0.0);
          goma_ctx->fem.pmv_hyst->cap_pres_switch[ipore] = alloc_dbl_1(exo->num_nodes, 0.0);
          goma_ctx->fem.pmv_hyst->sat_min_imbibe[ipore] = alloc_dbl_1(exo->num_nodes, 0.0);
          goma_ctx->fem.pmv_hyst->sat_max_drain[ipore] = alloc_dbl_1(exo->num_nodes, 0.0);
        }
      }
      break;
    }
  }

  /*
   * Stabilization_Variables__(if needed)___________________________________
   */
  for (mn = 0; mn < goma_ctx->fem.upd->Num_Mat && goma_ctx->fem.stab == NULL; mn++) {
    goma_ctx->fem.mp = goma_ctx->fem.mp_glob[mn];
    if (goma_ctx->fem.mp->Porous_wt_funcModel == SUPG ||
        goma_ctx->fem.vn_glob[mn]->wt_funcModel == SUPG ||
        goma_ctx->fem.mp->Spwt_funcModel == SUPG || goma_ctx->fem.mp->Mwt_funcModel == SUPG ||
        goma_ctx->fem.mp->Ewt_funcModel == SUPG) {
      goma_ctx->fem.stab = alloc_struct_1(STABILIZATION_PARAMS_STRUCT, 1);
      break;
    }
  }

  /*
   * Local_Element_Contributions___________________________________________
   */
  goma_ctx->fem.lec = alloc_struct_1(struct Local_Element_Contributions, 1);
  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Local_Element_Contributions @ %p has %ld bytes", yo,
            (void *)goma_ctx->fem.lec, (long int)sizeof(struct Local_Element_Contributions));
  }
  if (goma_ctx->fem.lec == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Local_Element_Contributions");
  }

  /*
   * Lubrication Auxiliaries_________________________________________________
   */
  goma_ctx->fem.lub_aux = alloc_struct_1(struct Lubrication_Auxiliaries, 1);
  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Lubrication_Auxiliaries @ %p has %ld bytes", yo,
            (void *)goma_ctx->fem.lub_aux, (long int)sizeof(struct Lubrication_Auxiliaries));
  }
  if (goma_ctx->fem.lub_aux == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Lubrication_Auxiliaries");
  }
  memset(goma_ctx->fem.lub_aux, 0, sizeof(struct Lubrication_Auxiliaries));

  goma_ctx->fem.lub_aux_old = alloc_struct_1(struct Lubrication_Auxiliaries, 1);
  if (goma_ctx->debug_flag) {
    DPRINTF(stdout, "%s: Lubrication_Auxiliaries Old @ %p has %ld bytes", yo,
            (void *)goma_ctx->fem.lub_aux_old, (long int)sizeof(struct Lubrication_Auxiliaries));
  }
  if (goma_ctx->fem.lub_aux_old == NULL) {
    status = -1;
    GOMA_EH(status, "Problem getting memory for Lubrication_Auxiliaries Old");
  }
  memset(goma_ctx->fem.lub_aux_old, 0, sizeof(struct Lubrication_Auxiliaries));

  return (status);
}
/***************************************************************************/
/***************************************************************************/
/***************************************************************************/

int bf_init(Exo_DB *exo)

/***********************************************************************
 * bf_init:
 *
 * Initialize some key pieces of each of the unique basis functions "bfd"
 * and set up the bf[variable] to point to the right "bfd"...
 *
 * For now, lets assume that bf[eqn] and bf[var] will be identical
 * references to the same basis function.
 *
 ***********************************************************************/
{
  int ebi, m, t = 0, v, status = 0;
  int imtrx;
  int el, shape, type;

  /*
   * For now, assume variable interpolations
   * and equation weightings are the same.
   *
   * We also assume that the interpolation for each variable is the
   * same for all materials in which it is active.
   *
   * Just make each var/eqn's bf point to the appropriate fundamental basis
   * function...
   */

  for (ebi = 0; ebi < goma_ctx->mesh.proc_num_elem_blk; ebi++) {
    m = goma_ctx->problem.matilda[ebi];
    if (m >= 0 && exo->eb_num_elems[ebi] > 0) {
      /* These are needed to check for matching element shapes */
      el = exo->eb_ptr[ebi];
      type = Elem_Type(exo, el);
      shape = type2shape(type);

      for (imtrx = 0; imtrx < goma_ctx->fem.upd->Total_Num_Matrices; imtrx++) {
        for (v = 0; v < MAX_VARIABLE_TYPES; v++) {
          /*
           * Is this variable active in the problem?
           */
          if (goma_ctx->fem.pd_glob[m]->v[imtrx][v]) {
            /*
             * If so, then check to see which prototype basis function of
             * bfd[] is its match...
             * NEW: Check both interpolation AND element shape!
             */
            for (t = 0; t < goma_ctx->problem.num_basis_functions; t++) {
              if (goma_ctx->fem.pd_glob[m]->i[imtrx][v] == goma_ctx->fem.bfd[t]->interpolation &&
                  shape == goma_ctx->fem.bfd[t]->element_shape) {
                if (goma_ctx->fem.bf[v] != 0) {
                  if (goma_ctx->fem.bf[v] != goma_ctx->fem.bfd[t]) {
                    GOMA_WH(-1, "bf[] struct isn't general enough to handle"
                                " variables with multiple interpolations in a single problem\n"
                                " We will carefully reset bf for each element");
                  }
                }
                goma_ctx->fem.bf[v] = goma_ctx->fem.bfd[t];
              }
            }
            if (goma_ctx->fem.bf[v] == NULL) {
              GOMA_EH(-1, "Could not find a match for variable.");
            }
          }
        }
      }
    }
  }

  return (status);
}

/*
 * Initialize some key pieces of each of the unique basis functions "bfd"
 * and set up the bf[variable] to point to the right "bfd"...
 *
 * For now, lets assume that bf[eqn] and bf[var] will be identical references
 * to the same basis function.
 *
 * Material specific matchup done.
 */
int bf_mp_init(struct Problem_Description *pd) {
  int ifound;
  int t, v;
  int status;
  int ishape;

  status = 0;

  t = 0;

  /* This is needed to check for matching element shapes */
  ishape = goma_ctx->fem.ei[goma_ctx->fem.pg->imtrx]->ielem_shape;

  /*
   * For now, assume variable interpolations
   * and equation weightings are the same.
   *
   * Just make each var/eqn's bf point to the appropriate fundamental basis
   * function...
   */
  for (v = 0; v < MAX_VARIABLE_TYPES; v++) {
    /*
     * Is this variable active in the material?
     *  If it isn't, then don't overwrite the entry, because we seem
     *  to need an idea for the interpolation even if its wrong in order
     *  for internal-boundary problems to work correctly.
     */
    int imtrx;
    for (imtrx = 0; imtrx < goma_ctx->fem.upd->Total_Num_Matrices; imtrx++) {
      if (pd->v[imtrx][v]) {
        /*
         * If so, then check to see which prototype basis function of
         * bfd[] is its match...
         * Check both interpolation AND element shape!
         */
        goma_ctx->fem.bf[v] = NULL;
        for (t = 0; t < goma_ctx->problem.num_basis_functions; t++) {
          if ((pd->i[imtrx][v] == goma_ctx->fem.bfd[t]->interpolation) &&
              (ishape == goma_ctx->fem.bfd[t]->element_shape))

            if ((pd->i[imtrx][v] == goma_ctx->fem.bfd[t]->interpolation) &&
                (ishape == goma_ctx->fem.bfd[t]->element_shape)) {
              goma_ctx->fem.bf[v] = goma_ctx->fem.bfd[t];
            }
        }
        if (goma_ctx->fem.bf[v] == NULL) {
          GOMA_EH(-1, "Could not find a match for variable.");
        }
      }
    }
  }
  /*
   * Repeat the same proceedure for external fixed field interpolations
   */

  /*
   * Are there external fields defined in the problem?
   */
  if (goma_ctx->fem.efv->ev) {
    for (v = 0; v < goma_ctx->fem.efv->Num_external_field; v++) {
      /*
       * If so, then check to see which prototype basis function of
       * bfd[] is its match...
       */
      ifound = 0;
      for (t = 0; t < goma_ctx->problem.num_basis_functions; t++) {
        if (goma_ctx->fem.efv->i[v] == goma_ctx->fem.bfd[t]->interpolation) {
          goma_ctx->fem.bfex[v] = goma_ctx->fem.bfd[t];
          ifound = 1;
        }
      }
      if (!ifound && goma_ctx->fem.efv->i[v] != I_TABLE) {
        GOMA_EH(GOMA_ERROR, "Could not find a match for EXTERNAL variable. Match some active field "
                            "interpolation with those of external variables");
      }
      if (goma_ctx->fem.bfex[v] == NULL && goma_ctx->fem.efv->i[v] != I_TABLE) {
        GOMA_EH(-1, "Could not find a match for EXTERNAL variable.");
      }
    }
  }

  return (status);
}

/*
 * Initialize the basis function representation to the NULL vector
 */
void bf_reset(void) {
  int v;
  for (v = 0; v < MAX_VARIABLE_TYPES; v++) {
    goma_ctx->fem.bf[v] = NULL;
  }
  if (goma_ctx->fem.efv->ev) {
    for (v = 0; v < goma_ctx->fem.efv->Num_external_field; v++) {
      goma_ctx->fem.bfex[v] = NULL;
    }
  }
}

/*
 * Initialize a freshly allocated structure to some safe defaults.
 *
 * Created: 2000/01/27 14:21 MST pasacki@sandia.gov
 *
 * Revised:
 */
static void init_Viscoelastic_Nonmodal(struct Viscoelastic_Nonmodal *v) {
  v->ConstitutiveEquation = 0;
  v->wt_func = (double)0;
  v->wt_funcModel = 0;
  v->eps = (double)0;
  v->evssModel = 0;
  v->modes = 0;
  v->dg_J_model = 0;
  v->dg_J_model_wt = NULL;
  v->len_dg_J_model_wt = 0;
  v->shiftModel = 0;
  v->shift = NULL;
  v->len_shift = 0;
  return;
}

/*
 * Initialize a freshly allocated structure to some safe defaults.
 *
 * Created: 2000/01/27 14:34 MST pasacki@sandia.gov
 *
 * Revised:
 */

static void init_Viscoelastic_Constitutive(struct Viscoelastic_Constitutive *v) {
  v->gn = NULL;
  v->time_const_st = NULL;
  v->alpha = (double)0;
  v->alphaModel = 0;
  v->xi = (double)0;
  v->xiModel = 0;
  v->eps = (double)0;
  v->epsModel = 0;
  v->pos_ls.alpha = 0.0;
  v->pos_ls.eps = 0.0;
  v->pos_ls.xi = 0.0;
  return;
}

/*
 * Initialize a freshly allocated structure to some safe defaults.
 *
 * Created: 2000/01/27 14:38 MST pasacki@sandia.gov
 *
 * Revised:
 */

static void init_Generalized_Newtonian(struct Generalized_Newtonian *g) {
  g->ConstitutiveEquation = 0;
  g->mu0 = (double)0;
  g->mu0Model = 0;
  g->nexp = (double)0;
  g->nexpModel = 0;
  g->muinf = (double)0;
  g->muinfModel = 0;
  g->lam = (double)0;
  g->lamModel = 0;
  g->aexp = (double)0;
  g->aexpModel = 0;
  g->atexp = (double)0;
  g->atexpModel = 0;
  g->wlfc2 = (double)0;
  g->wlfc2Model = 0;
  g->tau_y = (double)0;
  g->tau_yModel = 0;
  g->fexp = (double)0;
  g->fexpModel = 0;
  g->maxpack = (double)0;
  g->maxpackModel = 0;
  g->sus_species_no = 0;
  g->gelpoint = (double)0;
  g->gelpointModel = 0;
  g->cureaexp = (double)0;
  g->cureaexpModel = 0;
  g->curebexp = (double)0;
  g->curebexpModel = 0;
  g->tgel0 = (double)0;
  g->tgel0Model = 0;
  g->cure_species_no = 0;
  g->DilViscModel = 0;
  g->DilVisc0 = 0.0;

  return;
}

/*
 * Initialize a freshly allocated structure to some safe defaults.
 *
 * Created: 2000/01/27 14:49 MST pasacki@sandia.gov
 *
 * Revised:
 */

static void init_Elastic_Constitutive(struct Elastic_Constitutive *e) {
  int i;

  e->ConstitutiveEquation = 0;
  e->lame_mu = (double)0;
  e->lame_mu_model = 0;
  e->len_u_mu = 0;
  e->u_mu = NULL;
  for (i = 0; i < MAX_VARIABLE_TYPES + MAX_CONC; i++) {
    e->d_lame_mu[i] = (double)0;
  }
  e->lame_mu_tableid = 0;
  e->u_mu_ns = NULL;
  e->len_u_mu_ns = 0;
  e->multi_contact_line_distances = NULL;

  e->lame_lambda = (double)0;
  e->lame_lambda_model = 0;
  e->len_u_lambda = 0;
  e->u_lambda = NULL;
  for (i = 0; i < MAX_VARIABLE_TYPES + MAX_CONC; i++) {
    e->d_lame_lambda[i] = (double)0;
  }

  e->lame_TempShift = 0;
  e->lameTempShiftModel = 0;
  e->len_u_lame_TempShift = 0;
  e->u_lame_TempShift = NULL;
  for (i = 0; i < MAX_VARIABLE_TYPES + MAX_CONC; i++) {
    e->d_lame_TempShift[i] = 0;
  }
  e->lame_TempShift_tableid = 0;

  e->bend_stiffness = 0;
  e->bend_stiffness_model = 0;
  e->len_u_bend_stiffness = 0;
  e->u_bend_stiffness = NULL;

  for (i = 0; i < MAX_VARIABLE_TYPES + MAX_CONC; i++) {
    e->d_bend_stiffness[i] = 0;
  }

  e->poisson = (double)0;
  e->Strss_fr_sol_vol_frac = (double)0;

  for (i = 0; i < DIM; i++) {
    e->v_mesh_sfs[i] = (double)0;
  }

  e->v_mesh_sfs_model = 0;
  e->len_u_v_mesh_sfs = 0;
  e->u_v_mesh_sfs = NULL;

  e->thermal_expansion = (double)0;
  e->thermal_expansion_model = 0;
  e->len_u_thermal_expansion = 0;
  e->len_u_solid_viscosity = 0;
  e->len_u_solid_dil_viscosity = 0;
  e->len_u_solid_retardation = 0;
  e->u_thermal_expansion = NULL;
  e->solid_reference_temp = (double)0;
  e->solid_reference_temp_model = 0;

  return;
}

static void init_Viscoplastic_Constitutive(struct Viscoplastic_Constitutive *v) {
  int i;

  v->ConstitutiveEquation = 0;
  v->update_flag = 0;

  v->plastic_mu = (double)0;
  v->plastic_mu_model = 0;
  v->len_u_plastic_mu = 0;
  v->u_plastic_mu = NULL;
  for (i = 0; i < MAX_VARIABLE_TYPES + MAX_CONC; i++) {
    v->d_plastic_mu[i] = (double)0;
  }
  v->yield = (double)0;
  v->yield_model = 0;
  v->len_u_yield = 0;
  v->u_yield = NULL;
  for (i = 0; i < MAX_VARIABLE_TYPES + MAX_CONC; i++) {
    v->d_yield[i] = (double)0;
  }
  v->F_vp_glob = NULL;
  v->F_vp_old_glob = NULL;
  v->TT_glob = NULL;
  v->TT_old_glob = NULL;
  v->dTT_dx_glob = NULL;
  v->dTT_dx_old_glob = NULL;
  v->dTT_dc_glob = NULL;
  v->dTT_dc_old_glob = NULL;
  return;
}

static int shape_list(Exo_DB *exo)
/*
 * Determines number and type of element shapes in use.
 * Sets Num_Shapes and populates Unique_Shapes array.
 */
{
  int e, e1, e2, eshape, etype, there, N;

  /* Elements to search */
  e1 = exo->eb_ptr[0];
  e2 = exo->eb_ptr[exo->num_elem_blocks];

  /* Initialize shape list with first element */
  etype = Elem_Type(exo, e1);
  eshape = type2shape(etype);
  goma_ctx->problem.unique_shapes[0] = eshape;
  N = 1;

  /* Loop over all other elements, check shapes */
  for (e = e1 + 1; e < e2; e++) {
    etype = Elem_Type(exo, e);
    eshape = type2shape(etype);
    there = in_list(eshape, 0, N, goma_ctx->problem.unique_shapes);

    /* Add to list if not there */
    if (there == -1) {
      goma_ctx->problem.unique_shapes[N] = eshape;
      N++;
    }
  }

  if (N > 1)
    DPRINTF(stdout, "\nFound %d different element shapes.\n", N);
  return N;
}

/* end of file mm_as_alloc.c */
