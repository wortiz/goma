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
#include "rf_io_const.h"

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
} GomaContext;

/*
 * Lifecycle
 *
 * Memory owned by a context, and freed by goma_context_release():
 *   request, status, petsc_options, spec_source_inventory,
 *   spec_source_lumped_mass, mesh.coor (the pointer table only; the rows
 *   alias Exo_DB coordinates), mesh.proc_ss_node_count, and the particle
 *   arrays (number_samples*, number_output_variables, output_variables,
 *   filename_template, pbcs, element_info).
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

/* Free memory owned by the mesh / particle portions of a context. */
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
