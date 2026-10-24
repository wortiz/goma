/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) Goma Developers, National Technology & Engineering        *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

/*
 * Reset hooks called by goma_destroy (goma_api.c) so that Goma can be run
 * more than once in a process.
 *
 * Each <prefix>_free_globals() lives in the file that defines the globals
 * and statics it resets: it frees memory they own that is not freed
 * elsewhere and restores every one of them to its load time value.  When
 * adding a global or a function static that keeps state between calls, reset
 * it in its file's hook.
 */

#ifndef GOMA_RESET_H
#define GOMA_RESET_H

#ifdef __cplusplus
extern "C" {
#endif

void globals_free_globals(void);             /* globals.c */
void rf_util_free_globals(void);             /* rf_util.c */
void rd_mesh_free_globals(void);             /* rd_mesh.c */
void unknown_map_free_globals(void);         /* mm_unknown_map.c */
void bc_conflict_free_globals(void);         /* mm_bc_conflict.c */
void map_comm_vec_free_globals(void);        /* dp_map_comm_vec.c */
void vars_free_globals(void);                /* rf_vars.c */
void fill_util_free_globals(void);           /* mm_fill_util.c */
void fill_stabilization_free_globals(void);  /* mm_fill_stabilization.c */
void fill_ptrs_free_globals(void);           /* mm_fill_ptrs.c */
void fill_free_globals(void);                /* mm_fill.c */
void sl_util_free_globals(void);             /* sl_util.c */
void dp_utils_free_globals(void);            /* dp_utils.c */
void qtensor_free_globals(void);             /* mm_qtensor_model.c */
void bc_free_globals(void);                  /* mm_bc.c */
void eh_free_globals(void);                  /* mm_eh.c */
void input_free_globals(void);               /* mm_input.c */
void post_proc_free_globals(void);           /* mm_post_proc.c */
void sol_nonlinear_free_globals(void);       /* mm_sol_nonlinear.c */
void umf_free_globals(void);                 /* sl_umf.c */
void rf_solve_free_globals(void);            /* rf_solve.c */
void rf_solve_segregated_free_globals(void); /* rf_solve_segregated.c */
void lu_free_globals(void);                  /* sl_lu.c */
void luf_free_globals(void);                 /* sl_lu_fill.c */
void fill_solid_free_globals(void);          /* mm_fill_solid.c */
void fill_porous_free_globals(void);         /* mm_fill_porous.c */
void sl_aux_free_globals(void);              /* sl_aux.c */
void rotate_free_globals(void);              /* bc/rotate.c */
void rotate_coordinates_free_globals(void);  /* bc/rotate_coordinates.c, before rd_mesh */
void shell_util_free_globals(void);          /* mm_shell_util.c */
void fill_species_free_globals(void);        /* mm_fill_species.c */
void fill_ls_free_globals(void);             /* mm_fill_ls.c */
void loca_interface_free_globals(void);      /* ac_loca_interface.c */
void loca_util_free_globals(void);           /* loca_util.c */
void amesos_free_globals(void);              /* sl_amesos_interface.cpp */
void leastsquares_free_globals(void);        /* sl_leastsquares.cpp */
void stability_free_globals(void);           /* ac_stability.c */
void side_data_free_globals(void);           /* wr_side_data.c */
void pixel_image_free_globals(void);         /* rd_pixel_image.c */
void el_quality_free_globals(void);          /* el_quality.c */
void particles_free_globals(void);           /* ac_particles.c */
void eggrollwrap_free_globals(void);         /* sl_eggrollwrap.c */
void omega_h_free_globals(void);             /* adapt/omega_h_interface.cpp */
void loca_bord_free_globals(void);           /* loca_bord.c */
void ad_util_free_globals(void);             /* ad/util.cpp */
void stratimikos_free_globals(void);         /* sl_stratimikos_interface.cpp */
void vif_free_globals(void);                 /* dp_vif.c, needs upd and mp_glob: call early */
void assembly_free(void);                    /* mm_as_alloc.c, needs upd: call early */
void as_free_globals(void);                  /* mm_as_alloc.c, frees upd, pd_glob, ... */

/*
 * Frees the Amesos solvers, which refer to their matrix: call before
 * destroying ams->GomaMatrixData (sl_amesos_interface.cpp)
 */
void amesos_free_solvers(void);

#ifdef __cplusplus
}
#endif

#endif /* GOMA_RESET_H */
