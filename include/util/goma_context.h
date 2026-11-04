/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) 2026 Goma Developers, National Technology & Engineering   *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

#ifndef GOMA_UTIL_GOMA_CONTEXT_H
#define GOMA_UTIL_GOMA_CONTEXT_H

#include <mpi.h>
#include <stdio.h>

#include "ac_particles_types.h"
#include "el_elm.h"      /* DIM */
#include "mm_mp_const.h" /* MAX_NUMBER_MATLS */
#include "mm_variable_initialization.h"
#include "rf_bc_limits.h" /* MAX_MAT_PER_SS, MAX_*_TABLES */
#include "rf_fem_const.h" /* MAX_NUM_MATRICES, MAX_VARIABLE_TYPES */
#include "rf_io_const.h"  /* MAX_CHAR_IN_INPUT */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Legacy per-processor mesh description, formerly the globals in el_geom.h.
 * Deprecated in favor of the Exo_DB structure.
 */
typedef struct GomaMesh {
  double **coor;         /* physical space coordinates for each node, num_dim by num_node */
  int *proc_connect_ptr; /* global connectivity ptrs for element nums */

  int num_dim;     /* number of physical dimensions */
  int num_node;    /* total number of nodes in the mesh */
  int num_elem;    /* total number of elements in the mesh */
  int max_np_elem; /* maximum number of nodes in any element */

  int num_internal_elems; /* number of elements on the local processor ("internal" + "border") */
  int *gnodes;            /* internal, border and external nodes on each processor */
  int *gelems;            /* map from local to global element number, length num_internal_elems */

  int proc_num_elem_blk;     /* number of element blocks in this processor */
  int *proc_num_elem_in_blk; /* number of elements in the processor's element blocks */
  int *proc_elem_blk_ids;    /* element block id's for the processor's element blocks */
  int *proc_elem_blk_types;  /* element block types for the processor's element blocks */
  int *proc_nodes_per_elem;  /* # of nodes per element for each block on this processor */
  int *proc_num_attr;        /* # of attributes for each block on this processor */
  int *proc_elem_connect;    /* connectivity lists for the elements required by this processor */

  int proc_num_node_sets;    /* number node sets on current processor */
  int proc_ns_list_length;   /* total length of all the node set lists */
  int *proc_ns_ids;          /* node sets ids for the node sets */
  int *proc_ns_count;        /* node sets count record */
  int *proc_ns_pointers;     /* node sets pointer record */
  int *proc_ns_list;         /* node sets list record */
  double *proc_ns_dist_fact; /* node sets distribution factors */

  int proc_num_side_sets;       /* number side sets on current processor */
  int proc_ss_elem_list_length; /* total length of all the side set elem lists */
  int proc_ss_node_list_length; /* total length of all the side set node lists */
  int *proc_ss_ids;             /* side sets ids */
  int *proc_ss_elem_count;      /* side sets count record for elements */
  int *proc_ss_node_count;      /* side sets count record for nodes */
  int *proc_ss_elem_pointers;   /* side sets pointer record for elements */
  int *proc_ss_elem_list;       /* side sets element list record */
  double *proc_ss_dist_fact;    /* side sets distribution factors */

  int num_internal_nodes; /* owned exclusively by this proc */
  int num_border_nodes;   /* owned here but communicated elsewhere */
  int num_external_nodes; /* owned elsewhere, but needed here */
} GomaMesh;

/*
 * Particle tracking settings and state, formerly the globals in ac_particles.h.
 */
typedef struct GomaParticles {
  int dynamics;                                      /* particles are present */
  enum Particle_Model_t model;                       /* particle<->continuum coupling */
  double model_data[MAX_PARTICLE_MODEL_DATA_VALUES]; /* real values for this model */
  int number;                                        /* number of discrete particles */
  particle_filename_s restart_filename;              /* restart filename */
  int output_stride;                                 /* how often to output particle info */
  double output_time_step;                           /* output every these units */
  int max_time_steps;                          /* max particle time steps if steady solution */
  enum Particle_Output_Format_t output_format; /* what kind of output file */
  int full_output_stride;                      /* > 0 => full output every that many steps */
  particle_filename_s full_output_filename;    /* where to put full output */
  int number_sample_types;                     /* how many datasets to output */
  int *number_samples;                         /* particles to output for dataset #n */
  int *number_samples_existing;                /* particles tagged for each sample type */
  int *number_output_variables;                /* output vars for each sample */
  particle_variable_s **output_variables;      /* variables to output for dataset #n */
  particle_filename_s *filename_template;      /* template of where to put the data */

  double density;      /* density of particle in problem units */
  double radius;       /* radius of particle in problem units */
  double ratio;        /* real/computational particle ratio */
  int show_debug_info; /* show particle debug info */
  enum Particle_Domain_t creation_domain;
  enum Particle_Domain_t move_domain;
  particle_filename_s creation_domain_filename;
  particle_s creation_domain_name;
  particle_filename_s move_domain_filename;
  particle_s move_domain_name;
  double creation_domain_reals[MAX_DOMAIN_REAL_VALUES];
  double move_domain_reals[MAX_DOMAIN_REAL_VALUES];
  double xi_boundary_tolerances[3];

  int num_pbcs; /* number of particle-related sideset BC's */
  PBC_t *pbcs;  /* particle boundary condition structures */

  element_particle_info_t *element_info; /* per-element particle data, num_element_info long */
  struct Particle_Tracker *tracker;      /* particle lists, files and timers (ac_particles.c) */
  int num_element_info;
} GomaParticles;

struct Particle_Tracker;
struct Solve_Problem_State;
struct UMF_Linear_Solver_System;

#ifndef String_line_is_defined
#define String_line_is_defined
typedef char String_line[MAX_CHAR_IN_INPUT];
#endif

/*
 * Linear and nonlinear solver options, formerly the globals in rf_solver.h.
 * The Matrix_* strings are read from the input deck and interpreted later;
 * their meanings follow the Aztec manual (SAND 95-1559).
 */
typedef struct GomaSolver {
  String_line matrix_solver;
  String_line matrix_format;
  String_line matrix_scaling;
  String_line matrix_preconditioner;
  String_line matrix_subdomain_solver; /* Aztec 2.x */
  String_line matrix_residual_norm_type;
  String_line matrix_output_type;
  String_line matrix_factorization_reuse;
  String_line matrix_graph_fillin; /* Aztec 2.x */
  String_line matrix_maximum_iterations;
  String_line matrix_polynomial_order;
  String_line matrix_factor_overlap;
  String_line matrix_overlap_type;
  String_line matrix_krylov_subspace;
  String_line matrix_orthogonalization;
  String_line matrix_auxiliary_vector;
  String_line matrix_convergence_tolerance;
  String_line matrix_drop_tolerance;
  String_line matrix_factorization_save; /* Aztec 2 */
  String_line matrix_ilut_fill_factor;   /* Aztec 2 */
  String_line matrix_rilu_relax_factor;  /* Aztec 2 */
  String_line matrix_bilu_threshold;     /* Trilinos 1 */
  String_line matrix_relative_threshold; /* Trilinos 2 */
  String_line matrix_absolute_threshold; /* Trilinos 2 */
  String_line matrix_reorder;            /* Aztec 2.0 */
  String_line amesos_package;
  String_line amesos2_package;
  String_line aztecoo_solver;
  String_line stratimikos_file[MAX_NUM_MATRICES];
  String_line amesos2_file[MAX_NUM_MATRICES];

  int linear_solver; /* Aztec, Sparse, MA28, UMFPACK */
  int umfpack_idim;  /* UMFPACK storage constant */
  int umfpack_xdim;  /* UMFPACK storage constant */
  int loca_umf_id;   /* UMFPACK system id */

  int max_newton_steps;  /* maximum number of Newton steps to take */
  int guess_flag;        /* type of initial guess */
  int conformation_flag; /* mapping from stress to log-conformation tensor */
  int print3dbcdup;

  double damp_factor;
  double damp_factor1; /* Newton relaxation: 1.0 is full Newton, 0.0 does not update */
  double damp_factor2; /* additional damping factors and NORM(0,0) tolerances */
  double damp_factor3; /* for custom automatic control schemes */
  double custom_tol1;
  double custom_tol2;
  double custom_tol3;
  double var_damp[MAX_VARIABLE_TYPES]; /* variable specific damp factors */

  int newt_jacobian_reformation_stride; /* jacobian reformation stride, modified Newton */
  int time_jacobian_reformation_stride;
  int newton_line_search_type;
  double line_search_minimum_damping;
  int modified_newton;               /* modified Newton flag */
  int save_old_a;                    /* save old A matrix for Aztec resolves (see sl_util.c) */
  double convergence_rate_tolerance; /* jacobian reformation tolerance on convergence rate */
  double modified_newt_norm_tol;     /* jacobian reformation tolerance on residual norm */

  double epsilon[MAX_NUM_MATRICES][3]; /* stopping criteria */
  int solver_output_format;            /* bitmap for solver output columns */
  int output_variable_stats;           /* toggle for variable statistics */
  int output_variable_regression;      /* toggle for variable regression */

  int nzeros;      /* nonzeros in this proc's matrix */
  int gnzeros;     /* nonzeros in the global matrix */
  int fill_zeros;  /* nonzeros in the fill matrix for this processor */
  int gfill_zeros; /* nonzeros in the fill matrix for the global problem */

  int pspg;          /* pressure stabilized Petrov-Galerkin */
  int pspp;          /* pressure stabilized polynomial projection */
  double ps_scaling; /* constant scaling for the PSPG or PSPP term */
  int cont_gls;      /* continuity stabilization */

  int filter_species;
  int filter_species_material_number;
  double c_min;
  double c_max;

  int nnz_own;   /* nonzeroes in the matrix that belong to this processor */
  int nnz_total; /* nonzeroes this processor sees, including external unknowns */

  /* error and element-quality status flags, local and reduced over processors */
  int parallel_err;
  int parallel_err_global;
  int neg_elem_volume;
  int neg_elem_volume_global;
  int neg_lub_height;
  int neg_lub_height_global;
  int zero_detj;
  int zero_detj_global;

  int include_visc_sens; /* include viscosity sensitivities in the jacobian */
  int visc_sens_copy;
  int visc_sens_factor;

  /*
   * Solver state kept between calls. Each package allocates its own entries
   * and frees them in its release routine, called from goma_problem_release().
   */
  /* UMFPACK systems created by SL_UMF() (sl_umf.c), UMF_MAX_SYSTEMS long once allocated */
  struct UMF_Linear_Solver_System *umf_systems;
  int umf_num_systems; /* slots handed out so far; ids are not reused */

  /* SPARSE matrix factored by lu() (sl_lu.c), reused while the matrix index is unchanged */
  void *sparse_matrix;     /* the spCreate() matrix */
  void *sparse_elements;   /* spREAL ** element table for refilling it */
  int sparse_factored;     /* sparse_matrix has been ordered and factored */
  int sparse_matrix_imtrx; /* matrix index sparse_matrix belongs to */
  int sparse_calls;        /* lu() calls, for MATRIX_STATISTICS */

  /* row scaling work space of lu_decomp() (sl_aux.c) */
  double *lu_decomp_scaling;
  int lu_decomp_scaling_len;

  /* Trilinos solvers: amesos_solve() matrices and factorizations (struct Amesos_State in
   * sl_amesos_interface.cpp) and the trilinos_solve_ls() matrix (sl_leastsquares.cpp) */
  void *amesos;
  void *least_squares_matrix;

  /* nonlinear solver (mm_sol_nonlinear.c) */
  int first_linear_solve;    /* the next linear solve must set up the matrix */
  int nl_prev_matrix;        /* matrix index of the previous Newton solve */
  int nl_dofs_hidden;        /* external dofs are hidden from the linear solver */
  int nl_umf_system_id;      /* UMFPACK system of the Newton solve */
  int nl_factor_flag;        /* UMFPACK factorization option of the Newton solve */
  int eggroll_umf_system_id; /* UMFPACK system of the eggroll eigensolver */

  /* LOCA continuation interface (ac_loca_interface.c) */
  int loca_first_linear_solve; /* linear_solver_conwrap() */
  int loca_factor_flag;
  int loca_stab_first_linear_solve; /* shifted_linear_solver_conwrap() */
  int loca_stab_umf_id;
  int loca_stab_factor_flag;
  int loca_sv_init;    /* calc_scale_vec_conwrap() must allocate its arrays */
  int loca_step_print; /* solution_output_conwrap() output counters */
  int loca_n_print;
  int loca_eig_old_step; /* eigenvector_output_conwrap() output counters */
  int loca_eig_nprint;

  /* call counters and once-only flags */
  int sl_init_calls;     /* sl_init() calls (sl_util.c) */
  int matrix_dump_calls; /* matrix_dump() files written (sl_matrix_util.c) */
  int matrix_dump_msr_index;
  int matrix_dump_vbr_index;
  int stratimikos_tpetra_param_echo[MAX_NUM_MATRICES]; /* echo the solver parameters */
  int stratimikos_param_echo[MAX_NUM_MATRICES];

  /* solve_problem() state kept between calls (rf_solve.c) */
  struct Solve_Problem_State *solve_state;
} GomaSolver;

/* FEM data structures referenced from GomaFem; defined in mm_as_structs.h / mm_mp_structs.h */
struct AC_Information;
struct AD_Field_Variables;
struct Communication_Exchange;
struct Distributed_Processing_Information;
struct Element_Block_Struct;
struct Exodus_Database;
struct Extended_Shape_Fcn_Basics;
struct Nodal_Vars;
struct Node_Info;
struct Shape_Fcn_Tree;
struct Variable_Description;
struct Action_Flags;
struct Basis_Functions;
struct Constitutive_Relations;
struct Continuation_Conditions;
struct Continuation_Information;
struct Diet_Field_Variables;
struct Eigensolver_Info;
struct Elastic_Constitutive;
struct Element_Indices;
struct Element_Quality_Metrics;
struct Element_Stiffness_Pointers;
struct Element_Variable_Pointers;
struct External_Field_Variables;
struct Field_Variables;
struct Generalized_Newtonian;
struct HC_Information;
struct Level_Set_Data;
struct Level_Set_Interface;
struct Library_IO;
struct Local_Element_Contributions;
struct Loca_Input;
struct Lubrication_Auxiliaries;
struct Material_Properties;
struct Phase_Function_Data;
struct Porous_Media_Variables;
struct Porous_Media_Variables_Hysteresis;
struct Porous_Media_Variables_ML;
struct Problem_Description;
struct Problem_Graph;
struct Rotation_Vectors;
struct Shell_Block;
struct Stabilization_Params;
struct Transient_Information;
struct Uniform_Problem_Description;
struct User_Continuation_Info;
struct Viscoelastic_Constitutive;
struct Viscoelastic_Nonmodal;
struct Viscoplastic_Constitutive;

/*
 * Problem description, material properties and per-element working storage,
 * formerly global pointers defined in mm_as_alloc.c (declared in mm_as.h and mm_mp.h).
 *
 * The *_glob arrays are indexed by material number; pd, mp, cr, gn, ve, vn, elc,
 * elc_rs and evpl point at the entry for the current material.
 */
typedef struct GomaFem {
  /* Problem description */
  struct Uniform_Problem_Description *upd;
  struct Problem_Description **pd_glob, *pd;
  struct Problem_Graph *pg;
  struct Action_Flags *af;

  /* Material properties and constitutive models */
  struct Material_Properties *mp, **mp_glob, *mp_old;
  struct Constitutive_Relations **cr_glob, *cr;
  struct Generalized_Newtonian *gn, **gn_glob;
  struct Viscoelastic_Constitutive **ve, ***ve_glob;
  struct Viscoelastic_Nonmodal *vn, **vn_glob;
  struct Elastic_Constitutive *elc, **elc_glob, *elc_rs, **elc_rs_glob;
  struct Viscoplastic_Constitutive *evpl, **evpl_glob;

  /* Element indices, stiffness pointers and quality metrics */
  struct Element_Indices **ei;
  struct Element_Indices **ei_related;
  struct Element_Stiffness_Pointers *esp;
  struct Element_Variable_Pointers *esp_old, *esp_dot, *esp_dbl_dot, *evp;
  struct Element_Quality_Metrics *eqm;
  struct Local_Element_Contributions *lec;

  /* Basis functions */
  struct Basis_Functions **bf;   /* by variable type */
  struct Basis_Functions **bfd;  /* distinct basis functions */
  struct Basis_Functions **bfi;  /* by interpolation type */
  struct Basis_Functions **bfex; /* external fields */
  struct Shell_Block **shell_blocks;

  /* Field variables at the current quadrature point */
  struct Field_Variables *fv, *fv_sens;
  struct Diet_Field_Variables *fv_old, *fv_dot, *fv_dot_old, *fv_dot_dot, *fv_dot_dot_old;
  struct External_Field_Variables *efv;
  struct Porous_Media_Variables *pmv, *pmv_old;
  struct Porous_Media_Variables_Hysteresis *pmv_hyst;
  struct Porous_Media_Variables_ML *pmv_ml;
  struct Lubrication_Auxiliaries *lub_aux, *lub_aux_old;
  struct Stabilization_Params *stab;

  /* Level set / phase functions */
  struct Level_Set_Data *ls;
  struct Level_Set_Interface *lsi;
  struct Phase_Function_Data *pfd;

  /* Time integration, continuation, eigen and library I/O controls */
  struct Transient_Information *tran;
  struct Continuation_Information *cont;
  struct Continuation_Conditions *cpcc, *tpcc;
  struct User_Continuation_Info *cpuc, *tpuc;
  struct HC_Information *hunt;
  struct AC_Information *augc;
  struct Loca_Input *loca_in;
  struct Eigensolver_Info *eigen;
  struct Library_IO *libio;

  /* Rotated boundary condition vectors */
  struct Rotation_Vectors ****rotation;
  int **local_rot_list;
  int rotation_allocated;

  /* Element blocks and nodes */
  struct Element_Block_Struct *element_blocks; /* all element blocks */
  struct Element_Block_Struct *current_eb_ptr; /* block being processed in the fill */
  struct Node_Info **nodes;

  /* Solution vectors being assembled (set by the fill routines) */
  double *x_static;
  double *x_old_static;
  double *xdot_static;
  double *xdot_old_static;
  double *x_dbl_dot_static;
  double *x_dbl_dot_old_static;
  double *x_pred_static;
  double *p0; /* points at a harmless zero for unused dof pointers */

  /* XFEM and subgrid integration */
  struct Extended_Shape_Fcn_Basics *xfem;
  struct Shape_Fcn_Tree *subgrid_tree;

  /* Shells */
  int in_shell_element_with_parent_element_coverage;
  int shell_element_parent_element_coverage_for_variable[MAX_VARIABLE_TYPES];
  int **elem_friends;
  int *num_elem_friends;
  int num_shell_blocks;

  /* Q-tensor model scratch at the quadrature points of the current element */
  int mmh_ip;
  int prs_mat_ielem;
  double vort_dir[MDE][DIM];
  double qtensor[MDE][DIM][DIM];
  double div_qtensor[MDE][DIM];

  struct AD_Field_Variables *ad_fv; /* automatic differentiation field variables */
  double mm_fill_total;             /* time spent in matrix_fill */
} GomaFem;

/*
 * Problem setup: interpolation choices, time integration and continuation
 * parameters, augmenting condition counts, basis function inventory and the
 * unknown/DOF maps. Formerly the globals in rf_fem.h.
 */
typedef struct GomaProblem {
  int coordinate_system; /* type of coordinate system (see rf_fem_const.h) */

  /* FEM interpolation parameters (see rf_fem_const.h) */
  int velocity_pressure; /* element type for velocity and pressure interpolation */
  int velocity;          /* interpolation for velocity (set from velocity_pressure) */
  int pressure;          /* interpolation for pressure (set from velocity_pressure) */
  int temperature;       /* interpolation for temperature */
  int mesh_displacement; /* element type for mesh displacement interpolation */
  int mass_fraction;     /* interpolation for mass fraction and density */
  int n_eqm;             /* one or more element quality metrics are computed */
  int use_dg;            /* discontinuous Galerkin interpolation is in use */
  int do_overlap;        /* overlap augmenting conditions are used */
  int problem_type;      /* type of problem to be solved */

  /* extra terms in axisymmetric or swirling flow problems; set in setup_pd */
  int vim;
  int wim;

  /* boundary conditions, rotations and interface sources */
  int num_bc;
  int num_interface_srcs;
  int int_src_bc_id[MAX_INTERFACE];
  int num_rot;

  /* import / export field counts (coupled mode) */
  int num_import_nv;
  int num_import_ev;
  int num_export_xs;
  int num_export_xp;
  int export_xs_id[MAX_EXTERNAL_FIELD];
  int export_xp_id[MAX_EXTERNAL_FIELD];

  /* unique basis functions, element types, interpolations and shapes */
  int num_basis_functions;
  int unique_basis_functions[MAX_BASIS_FUNCTIONS];
  int num_element_types;
  int unique_element_types[MAX_ELEMENT_TYPES];
  int num_interpolations;
  int unique_interpolations[MAX_INTERPOLATIONS];
  int highest_interpolation;
  int num_shapes;
  int unique_shapes[MAX_ELEMENT_SHAPES];

  /* time integration */
  int time_integration;
  int use_level_set;
  int use_phase_field;
  double eps; /* time step error */
  int print_freq;
  double print_delt;
  double print_delt2_time;
  double print_delt2;

  /* continuation */
  int continuation;
  int cont_type;
  int bdy_cond_id;
  int data_flt_id;
  int mat_id;
  int mat_prop_id;
  int mat_prop_si_id;
  int max_path_steps;
  double delta_s0;
  double delta_s_min;
  double delta_s_max;
  double path_max;
  double print_delt2_path;
  double beg_parameter_value;
  double end_parameter_value;

  /* augmenting, continuation and hunting condition counts */
  int n_ac;
  int n_cc;
  int n_tc;
  int n_uc;
  int n_utc;
  int n_hc;

  int *matilda; /* material index for each element block; filled in rd_mesh.c */

  /* DOF counts by ownership: internal (mine only), boundary (mine, others want),
   * external (theirs, I want), universe (all known to this processor) */
  int *num_internal_dofs;
  int *num_boundary_dofs;
  int *num_external_dofs;
  int *num_universe_dofs;
  int num_personal_elems;

  /* communication lists for nodes and DOFs (dp_map_comm_vec.c) */
  int *ptr_node_recv;
  int *ptr_fill_node_recv;
  int **ptr_dof_send;
  int **list_dof_send;
  int *ptr_node_send;
  int *list_node_send;
  int *ptr_fill_node_send;
  int *list_fill_node_send;

  int *num_unknowns;     /* unknowns updated by this processor, per matrix */
  int *num_ext_unknowns; /* external unknowns stored on this processor, per matrix */
  int max_var_per_node;  /* maximum number of unknowns at any node on any processor */
  int num_var_in_type[MAX_NUM_MATRICES][MAX_VARIABLE_TYPES];

  int num_fill_unknowns;
  int owned_fill_unknowns;
  int internal_fill_unknowns;
  int boundary_fill_unknowns;
  int external_fill_unknowns;

  int ***local_offset; /* [matrix][node][variable] offset from the node's first unknown */
  int ***dolphin;      /* [matrix][node][variable] degrees of freedom (mm_unknown_map.c) */
  int ***idv;          /* [matrix][dof][0..2]: variable, nodal dof, global node */
  char ***dofname;     /* variable names for each dof */
  char ***resname;     /* residual equation names for each dof */
  int **first_unknown; /* [matrix][node] index of the node's first unknown */
  int inter_mask[MAX_NUM_MATRICES][MAX_VARIABLE_TYPES][MAX_VARIABLE_TYPES];
  int ignore_deps[MAX_NUM_MATRICES][MAX_VARIABLE_TYPES][MAX_VARIABLE_TYPES];
  int *node_index_fill;
  int *fill_node_list;

  /* variable descriptions and nodal variable layouts (rf_vars.c) */
  struct Variable_Description **var_info;
  int num_var_info_records;
  struct Nodal_Vars **nodal_vars_list;
  int nodal_vars_list_length;

  /* global and per-material variable initialization */
  struct Variable_Initialization var_init[MAX_VARIABLE_TYPES + MAX_CONC];
  struct Variable_Initialization var_init_mat[MAX_NUMBER_MATLS][MAX_VARIABLE_TYPES + MAX_CONC];
  int num_var_init_mat[MAX_NUMBER_MATLS];

  /* linear stability analysis */
  int linear_stability;  /* kind of linear stability, if any (LSA_* in ac_stability.h) */
  int lsa_3d_of_2d_pass; /* pass # when constructing the jacobian/mass matrices */
  double lsa_3d_of_2d_wave_number;
  double *lsa_wave_numbers;    /* wave numbers to compute (not necessarily integers) */
  int lsa_number_wave_numbers; /* length of lsa_wave_numbers */
  int lsa_current_wave_number; /* index of the current wave number */

  int chemkin_needed; /* some functionality needs the chemkin libraries */
} GomaProblem;

struct BC_descriptions;
struct Boundary_Condition;
struct Data_Table;
struct Rotation_Specs;
struct elem_edge_bc_struct;
struct elem_side_bc_struct;

/*
 * Boundary condition, rotation and data table state read from the input deck
 * and set up for the mesh. Formerly the globals in rf_bc.h and rf_bc_const.h.
 */
typedef struct GomaBC {
  struct Boundary_Condition *bc_types; /* boundary conditions from the input deck, num_bc long */
  struct Rotation_Specs *rot_types;    /* rotation specifications, num_rot long */
  int **rot_list;
  int **ss_list;
  int use_2d_rotation_vectors;

  /* per element: list of integrated BCs on sides / edges; NULL if none */
  struct elem_side_bc_struct ***first_elem_side_bc_array;
  struct elem_edge_bc_struct ***first_elem_edge_bc_array;

  /* dynamically allocated BC descriptions (sent to other processors) */
  int num_new_bc_desc;
  struct BC_descriptions **new_bc_desc;

  /* data tables attached to BCs, material properties, external fields and ACs */
  int num_bc_tables;
  struct Data_Table *bc_tables[MAX_BC_TABLES];
  int num_mp_tables;
  struct Data_Table *mp_tables[MAX_MP_TABLES];
  int num_ext_tables;
  struct Data_Table *ext_tables[MAX_EXT_TABLES];
  int num_ac_tables;
  struct Data_Table *ac_tables[MAX_AC_TABLES];

  /* BC conflict resolution and rotation bookkeeping (mm_bc_conflict.c) */
  int **bc_dup_nodes;
  int ****bc_dup_list;
  int *bc_dup_ptr;
  int *ss_to_blks[MAX_MAT_PER_SS + 1];
  int dup_blks_list[MAX_MAT_PER_SS + 1];
  int *ss_internal_boundary;
  int **mesh_rotate_node;
  int **mesh_rotate_ss;
  int *num_mesh_rotate;
  int **mom_rotate_node;
  int **mom_rotate_ss;
  int *num_mom_rotate;

  int pressure_datum;          /* a pressure datum is set */
  int pressure_datum_element;  /* element in which the pressure datum is set */
  double pressure_datum_value; /* value of the pressure datum */
} GomaBC;

struct Post_Processing_Averages;
struct Post_Processing_Data;
struct Post_Processing_Data_Sens;
struct Post_Processing_Error;
struct Post_Processing_Fluxes;
struct Post_Processing_Fluxes_Sens;
struct Post_Processing_Global;
struct Post_Processing_Particles;
struct Post_Processing_Volumetric;

/*
 * Post-processing requests and variable flags, formerly the globals in
 * mm_post_def.h / mm_post_proc.c.
 */
typedef struct GomaPost {
  /* requested fluxes, data, error metrics, particles, volume integrals, globals, averages */
  struct Post_Processing_Fluxes **pp_fluxes;
  struct Post_Processing_Fluxes_Sens **pp_fluxes_sens;
  struct Post_Processing_Data **pp_data;
  struct Post_Processing_Data_Sens **pp_data_sens;
  struct Post_Processing_Error *pp_error_data;
  struct Post_Processing_Particles **pp_particles;
  struct Post_Processing_Volumetric **pp_volume;
  struct Post_Processing_Global **pp_global;
  struct Post_Processing_Averages **pp_average;
  /* number of entries in each list above */
  int nn_post_fluxes;
  int nn_post_fluxes_sens;
  int nn_post_data;
  int nn_post_data_sens;
  int nn_error_metrics;
  int nn_particles;
  int nn_volume;
  int ppvi_type;
  int nn_global;
  int nn_average;

  int num_nodal_post_proc_var;
  int num_elem_post_proc_var;
  int len_u_post_proc; /* length of u_post_proc */
  double *u_post_proc; /* user-provided values for user-defined post processing */

  /* Post-processing variable flags: -1 if not requested, otherwise the id of the
   * (first) post-processing variable for this quantity, set in load_nodal_tkn. */
  int capillary_pressure;
  int conc_cont;
  int conduction_vectors;
  int curl_v;
  int darcy_velocity_gas;
  int darcy_velocity_liq;
  int density;
  int polymer_viscosity;
  int polymer_time_const;
  int mobility_parameter;
  int ptt_xi;
  int ptt_epsilon;
  int dielectrophoretic_field;
  int dielectrophoretic_field_norm;
  int enormsq_field;
  int enormsq_field_norm;
  int diffusion_vectors;
  int diffusion_vectors_por_liq_gphase;
  int diffusion_vectors_por_air_gphase;
  int div_pvelocity;
  int div_total;
  int div_velocity;
  int electric_field;
  int electric_field_mag;
  int energy_fluxlines;
  int error_zz_p;
  int error_zz_p_elsize;
  int error_zz_q;
  int error_zz_q_elsize;
  int error_zz_vel;
  int error_zz_vel_elsize;
  int evp_def_grad_tensor;
  int external_post;
  int fill_cont;
  int first_invar_strain;
  int fluxlines;
  int lagrange_convection;
  int mean_shear;
  int mm_residuals;
  int ns_residuals;
  int porous_rho_gas_solvents;
  int porous_rho_lphase;
  int porous_rho_total_solvents;
  int porous_saturation;
  int porous_gridpeclet;
  int porous_supgvelocity;
  int porous_liquid_accum_rate;
  int rel_liq_perm;
  int pressure_cont;
  int sh_div_s_v_cont;
  int sh_curv_cont;
  int real_stress_tensor;
  int sec_invar_strain;
  int strain_tensor;
  int stream;
  int stream_normal_stress;
  int stream_shear_stress;
  int stream_tension;
  int stress_cont;
  int stress_tensor;
  int surface_vectors;
  int shell_normals;
  int third_invar_strain;
  int time_derivatives;
  int total_stress11;
  int total_stress12;
  int total_stress13;
  int total_stress22;
  int total_stress23;
  int total_stress33;
  int user_post;
  int pp_viscosity;
  int pp_flowingliquid_viscosity;
  int pp_volumefractiongas;
  int sat_curve_type;
  int cap_press_switch;
  int sat_qp_switch;
  int num_curve_switch;
  int sat_hyst_min;
  int sat_hyst_max;
  int acoustic_pressure;
  int acoustic_phase_angle;
  int acoustic_energy_density;
  int light_intensity;
  int principal_stress;
  int principal_real_stress;
  int lub_height;
  int lub_height_2;
  int lub_velo_upper;
  int lub_velo_lower;
  int lub_velo_field;
  int lub_velo_field_2;
  int disj_press;
  int sh_sat_open;
  int sh_sat_open_2;
  int sh_cap_pres;
  int sh_pore_flux;
  int sh_stress_tensor;
  int sh_tang;
  int pp_lame_mu;
  int pp_lame_lambda;
  int von_mises_strain;
  int von_mises_stress;
  int conf_map;
  int j_flux;
  int eig;
  int eig1;
  int eig2;
  int eig3;
  int heaviside;
  int rho_dot;
  int moment_sources;
  int yzbeta;
  int grad_y;
  int grad_sh;
  int untracked_spec;
  int tfmp_gas_velo;
  int tfmp_liq_velo;
  int tfmp_inv_peclet;
  int tfmp_krg;
  int velo_speed;
  int gies_crit;
  int helicity;
  int lamb_vector;
  int q_fcn;
  int poynting_vectors;
  int pspg_pp;
  int saramito_yield;
  int stress_norm;
  int species_sources;
  int viscous_stress;
  int pp_velocity_gradients;
  int viscous_stress_norm;
  int viscous_von_mises_stress;
  int em_contours;
  int total_em_contours;
  int scattered_em_contours;
  int orientation_vectors;
  int first_strainrate_invar;
  int sec_strainrate_invar;
  int third_strainrate_invar;
  int wall_distance;
  int contact_distance;
  int pp_fluid_stress;
  int lub_convection;
  int pp_mesh_velocity;
  int lub_fluid_source;
  int mean_vorticity;
  int pp_diffusivity;
} GomaPost;

/*
 * GomaContext collects process-wide state that used to live in individual
 * global variables: MPI/parallel information and I/O / run configuration.
 *
 * A single instance is owned by goma_context.c and reachable through
 * goma_ctx, which is never NULL.
 */
/*
 * State that assembly, level set and output routines keep between calls
 * (first-call flags, reference values, open log files). goma_context_release()
 * closes the files and puts these back to their defaults.
 */
typedef struct GomaFillState {
  /* level set (mm_fill_ls.c) */
  double ls_simplemass_mass; /* reference level set mass, Hrenorm_simplemass() */
  int ls_simplemass_mass_set;
  double ls_smolianski_mass; /* reference level set mass, Hrenorm_smolianksi_only() */
  int ls_smolianski_mass_set;
  FILE *ls_surf_log;           /* level_set_all.dat, print_surf_list() */
  int ca_boundary_num_ss;      /* contact-angle sidesets found, -1 before the search */
  int ca_boundary_ss[20];      /* their ids, point_on_ca_boundary() */
  int subelement_output_calls; /* subelements.NNNNN.exoII files written */

  /* assembly first-call flags */
  int elem_ptrs_initialized; /* element index arrays reset once, mm_fill_ptrs.c */
  int porous_initialized;    /* load_porosity() (mm_fill_porous.c) */
  int dofptr_initialized;    /* mm_fill_util.c, for dofptr_elem_blk_id */
  int dofptr_elem_blk_id;
  int belly_flop_initialized;  /* mm_fill_solid.c */
  int pspg_initialized;        /* calc_pspg() */
  int cont_gls_initialized;    /* calc_cont_gls() */
  int mesh_derivs_initialized; /* load_fv_mesh_derivs() */
  int element_quality_called;  /* element_quality() has run once */
  int truth_table_calls;       /* create_truth_table*() calls that printed their warnings */
  int pixel_mesh_written;      /* rd_pixel_image.c copied the mesh for mapped fields */
  int pixel2_mesh_written;     /* rd_pixel_image2.c copied the mesh for mapped fields */
} GomaFillState;

/* Echo files of the input deck kept open by handle_echo_string() (mm_input.c). */
typedef struct GomaEcho {
  FILE *files[FOPEN_MAX];
  char *names[FOPEN_MAX]; /* file name of each open echo file */
  int num_files;
  int off; /* NOECHO was requested */
} GomaEcho;

typedef struct GomaContext {
  /* Parallel machine description and MPI communication bookkeeping */
  int num_proc;                       /* Total number of processors */
  int proc_id;                        /* This processor's number */
  int dim;                            /* Dimension of logical hypercube */
  MPI_Request *request;               /* Nonblocking send/recv requests */
  MPI_Status *status;                 /* Nonblocking send/recv statuses */
  int num_requests;                   /* Number of entries in request/status */
  int unlimited_output;               /* flag to unlimit multiprocessor output */
  struct Communication_Exchange **cx; /* communications info for each neighbor processor */

  /* Mesh database and distributed processing information */
  struct Exodus_Database *exo;
  struct Distributed_Processing_Information *dpi;

  /* Program invocation */
  int argc;
  char **argv;
  double time_goma_started;
  int number_jac_dump; /* number of jacobians to dump (MATRIX_DUMP builds) */

  /* Input / output files */
  char input_file[MAX_FNL];          /* input EXODUS II database w/ problem defn */
  char exo_file[MAX_FNL];            /* input EXODUS II database w/ problem defn */
  char exo_lb_file[MAX_FNL];         /* EXODUS II load balance info for mesh */
  char exo_file_out[MAX_FNL];        /* output EXODUS II database w/ results */
  char exo_file_out_mono[MAX_FNL];   /* output EXODUS II database without per proc string */
  char exo_aux_file[MAX_FNL];        /* auxiliary EXODUS II database for initguess */
  char echo_input_file[MAX_FNL];     /* echo of problem def file */
  char domain_mapping_file[MAX_FNL]; /* Maps materials and material boundary names into chemkin
                                        domains and surface/volumetric domains */
  char init_guess_file[MAX_FNL];     /* ASCII file holding initial guess */
  char soln_out_file[MAX_FNL];       /* ASCII file holding solution vector */
  int exo_time_plane;                /* Time plane # or continuation # of soln for initial guess */
  int cpu_word_size;                 /* Precision variables for exodus II files */
  int io_word_size;

  /* Run configuration */
  int decompose_flag; /* Flag to check for built-in brking */
  int decompose_type;
  int skip_fix;
  int debug_flag;                   /* Level of debug info to stdout: 0 none, 1 minimum, 2 medium,
                                       3 maximum, -1 check jacobian, -2 check jacobian w/ scaling */
  int iout;                         /* Level of diagnostic output for the program */
  int new_parser_flag;              /* 0: old parser, 1: flex/bison parser */
  int write_intermediate_solutions; /* Write solution data at each Newton iteration */
  int write_initial_solution;       /* Write initial solution to ascii and exodus output */
  int anneal_mesh;                  /* Create exodus file with coordinates adjusted to the
                                       deformed coordinates (new displacements set to zero) */
  char *petsc_options;              /* PETSc options string */
  int petsc_options_str_len;

  /* Initialization counts */
  int num_var_init;         /* Number of variables to overwrite with global initialization */
  int num_var_bound;        /* Number of variables to apply bounds */
  int num_var_ls_init;      /* Number of variables to overwrite with level set initialization */
  int num_var_external;     /* Number of external variables (exoII or pixel) */
  int num_var_external_pix; /* Number of external variables (pixel only) */

  /* Accumulated boundary condition / source data */
  double porous_liq_inventory;     /* finite-insult boundary condition */
  double **spec_source_inventory;  /* cumulative reacted source */
  int spec_source_inventory_rows;  /* number of rows allocated in spec_source_inventory */
  double *spec_source_lumped_mass; /* species lumped mass */

  /* Benner's frontal solver strcat()s onto this directory name, so leave room for
   * dirname/lu.123456.0 */
  char front_scratch_directory[MAX_FNL];

  GomaMesh mesh;           /* legacy mesh description */
  GomaParticles particles; /* particle tracking */
  GomaSolver solver;       /* linear and nonlinear solver options */
  GomaFem fem;             /* problem description, materials and element working storage */
  GomaProblem problem;     /* problem setup, continuation parameters and unknown maps */
  GomaBC bc;               /* boundary conditions, rotations and data tables */
  GomaPost post;           /* post-processing requests and variable flags */
  GomaFillState fill;      /* state assembly and output routines keep between calls */
  GomaEcho echo;           /* input deck echo files */

  /*
   * Frees what the solver library allocated through this context: the input
   * deck, problem setup, Exodus/decomposition databases and the fem
   * structures. Installed by the library that allocates them
   * (goma_problem_release() in rf_release.c, set by pd_alloc()) and called by
   * goma_context_release() with ctx current.
   */
  void (*release_problem)(struct GomaContext *ctx);
} GomaContext;

/*
 * Lifecycle
 *
 * Memory owned by a context, and freed by goma_context_release():
 *   request, status, petsc_options, spec_source_inventory,
 *   spec_source_lumped_mass, mesh.coor (the pointer table only; the rows
 *   alias Exo_DB coordinates), mesh.proc_ss_node_count, and the particle
 *   arrays (number_samples*, number_output_variables, output_variables,
 *   filename_template, pbcs, element_info), and through release_problem
 *   everything the solver library allocated: input deck data (materials,
 *   boundary conditions, data tables, continuation and augmenting
 *   conditions, level set, post-processing requests), problem setup (unknown
 *   and communication maps, nodes, element blocks, BC lists), exo, dpi, and
 *   the problem description and element structures from mm_as_alloc.c.
 *
 * All other mesh pointers alias Exo_DB storage and are only cleared.
 */

/* Set every field of ctx to the Goma startup default (GOMA_CONTEXT_DEFAULTS). Frees nothing. */
void goma_context_init(GomaContext *ctx);

/* Allocate a context initialized with goma_context_init(). Returns NULL on failure. */
GomaContext *goma_context_create(void);

/* Free all memory owned by ctx and clear the corresponding pointers. Settings are kept. */
void goma_context_release(GomaContext *ctx);

/* goma_context_release() followed by goma_context_init(). */
void goma_context_reset(GomaContext *ctx);

/*
 * Release and free a context from goma_context_create(). If ctx is the current
 * context, the built-in default context becomes current. NULL is ignored.
 */
void goma_context_destroy(GomaContext *ctx);

/* Free memory owned by the mesh / particle portions of a context. */
void goma_mesh_release(GomaMesh *mesh);
/* Close the files in fill and put it back to its defaults. */
void goma_fill_release(GomaFillState *fill);
/* Close the echo files and forget them. */
void goma_echo_release(GomaEcho *echo);
void goma_particles_release(GomaParticles *particles);
void goma_particles_release_element_info(GomaParticles *particles);

/*
 * Make ctx the context used through goma_ctx and return the previous one.
 * Passing NULL selects the built-in default context.
 */
GomaContext *goma_context_set_current(GomaContext *ctx);

/* The current context. Never NULL. */
extern GomaContext *goma_ctx;

#ifdef __cplusplus
}
#endif

#endif /* GOMA_UTIL_GOMA_CONTEXT_H */
