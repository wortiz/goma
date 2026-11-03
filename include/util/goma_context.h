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

#include "ac_particles_types.h"
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
  int num_element_info;
} GomaParticles;

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

  int include_visc_sens; /* include viscosity sensitivities in the jacobian */
  int visc_sens_copy;
  int visc_sens_factor;
} GomaSolver;

/* FEM data structures referenced from GomaFem; defined in mm_as_structs.h / mm_mp_structs.h */
struct AC_Information;
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

  /*
   * Frees the structures above. Installed by the library that allocates them
   * (fem_structures_free() in mm_as_alloc.c) and called by goma_context_release().
   */
  void (*release)(struct GomaFem *fem);
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

/*
 * GomaContext collects process-wide state that used to live in individual
 * global variables: MPI/parallel information and I/O / run configuration.
 *
 * A single instance is owned by goma_context.c and reachable through
 * goma_ctx, which is never NULL.
 */
typedef struct GomaContext {
  /* Parallel machine description and MPI communication bookkeeping */
  int num_proc;         /* Total number of processors */
  int proc_id;          /* This processor's number */
  int dim;              /* Dimension of logical hypercube */
  MPI_Request *request; /* Nonblocking send/recv requests */
  MPI_Status *status;   /* Nonblocking send/recv statuses */
  int num_requests;     /* Number of entries in request/status */
  int unlimited_output; /* flag to unlimit multiprocessor output */

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
} GomaContext;

/*
 * Lifecycle
 *
 * Memory owned by a context, and freed by goma_context_release():
 *   request, status, petsc_options, spec_source_inventory,
 *   spec_source_lumped_mass, mesh.coor (the pointer table only; the rows
 *   alias Exo_DB coordinates), mesh.proc_ss_node_count, and the particle
 *   arrays (number_samples*, number_output_variables, output_variables,
 *   filename_template, pbcs, element_info), and through fem.release the
 *   problem description, material and element structures from mm_as_alloc.c.
 *
 * All other mesh pointers alias Exo_DB storage and are only cleared.
 */

/* Set every field of ctx to the Goma startup default. Does not free anything. */
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

/* Free memory owned by the mesh / particle / fem portions of a context. */
void goma_fem_release(GomaFem *fem);
void goma_mesh_release(GomaMesh *mesh);
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
