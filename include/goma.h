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

/*
 * Library interface to Goma
 *
 * Typical use, with MPI initialized and finalized by the caller:
 *
 *   goma_library_initialize(MPI_COMM_WORLD);
 *   goma_context *ctx;
 *   goma_create(&ctx, MPI_COMM_WORLD, argc, argv);  // command line + input deck
 *   goma_setup(ctx);                                 // mesh and problem setup
 *   goma_solve(ctx);
 *   goma_destroy(&ctx);
 *   goma_library_finalize();
 *
 * Only one goma_context may exist at a time.  goma_destroy frees the problem
 * and resets Goma's global state, so a later goma_create can run another
 * (independent) problem.
 *
 * Goma runs on a duplicate of the communicator given to goma_create, which
 * may hold any subset of the processes.  Fatal errors still terminate the
 * process.
 *
 * All functions return 0 on success and a negative value on failure.
 */

#ifndef GOMA_H
#define GOMA_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct goma_context goma_context;

/* Call once per process after MPI_Init, before any other goma_* function,
 * collective over comm, which holds every process that may run a problem.
 * Initializes PETSc on a duplicate of comm unless the host initialized it. */
int goma_library_initialize(MPI_Comm comm);

/* Finalizes third-party libraries initialized by Goma (never MPI),
 * collective over the communicator of goma_library_initialize */
int goma_library_finalize(void);

/* Parse the command line (argv[0] is the program name) and input deck,
 * collective over comm */
int goma_create(goma_context **ctx, MPI_Comm comm, int argc, char **argv);

/*
 * Declare an imported external field (see goma_set_external_field) without
 * an External Field card in the deck.  Call before goma_create, with the same
 * arguments on all ranks; the declaration applies to the next problem only
 * and is the same as adding the card
 *
 *   External Field = name interpolation IMPORT      (GOMA_NODAL)
 *   External Field = name interpolation IMPORT_EV   (GOMA_ELEMENT)
 *
 * after the deck's External Field cards, so material property cards can
 * use the field.  interpolation is that of the card, Q1 or Q2 for example.
 * Returns an error for a name longer than 20 characters or declared twice,
 * an unknown interpolation, or too many external fields; declaring a field
 * that the deck also declares is a fatal error when the deck is read.
 */
typedef enum { GOMA_NODAL, GOMA_ELEMENT } goma_field_location;
int goma_declare_external_field(const char *name,
                                const char *interpolation,
                                goma_field_location location);

/* Read the mesh and set up the problem */
int goma_setup(goma_context *ctx);

/* Solve the problem, writing results to the output files.  After
 * goma_advance_*, solves the rest of the problem. */
int goma_solve(goma_context *ctx);

/*
 * Advance the problem in steps instead of goma_solve, for coupling with
 * another code.  Problems with a single matrix (steady or transient, no
 * continuation, hunting or LOCA) and problems solved by the segregated solver
 * (several matrices) can be advanced; the first call returns an error for
 * others.
 *
 * goma_advance_steps takes num_steps (>= 1) successful time steps.
 * goma_advance_to steps until the simulation time reaches time, shortening
 * the last step to land on it.  A steady problem is solved by the first
 * call.  Output is written at the deck's print frequency.  The time step
 * size carries over from one call to the next.
 *
 * Returns 0 when the problem can be advanced further, 1 when it has reached
 * its end (maximum time or steps, steady state) and a negative value on
 * failure.
 */
int goma_advance_steps(goma_context *ctx, int num_steps);
int goma_advance_to(goma_context *ctx, double time);

/* Simulation time of the last converged step and the size of the next
 * time step, both 0 for a steady problem.  Valid after goma_advance_*. */
int goma_get_time(goma_context *ctx, double *time, double *delta_t);

/*
 * Nodal values at this rank's nodes (num_nodes of goma_mesh_get_info) of the
 * solution after goma_advance_*, as they would be written to the output
 * file.  goma_get_field takes a variable name (VELOCITY1 or VX, MASS_FRACTION
 * or Y with the species number as component, 0 otherwise) or a nodal
 * variable name of the output file (a variable name wins over an output
 * name, PRESSURE is P).  goma_get_post_field takes a nodal post
 * processing variable requested in the deck, named as in the output file; it
 * computes the post processing variables and must be called on all ranks.
 * Both return an error if the problem has no such variable.
 */
int goma_get_field(goma_context *ctx, const char *name, int component, double *values);
int goma_get_post_field(goma_context *ctx, const char *name, double *values);

/*
 * Values of an external field declared with goma_declare_external_field or
 * in the deck with IMPORT (nodal values, GOMA_NODAL) or IMPORT_EV (element
 * values, GOMA_ELEMENT) in place of a file name, for example
 *
 *   External Field = SOURCE Q1 IMPORT
 *
 * values[num_nodes] or values[num_elems] of goma_mesh_get_info.  Only the
 * owned nodes' values are used, Goma copies them to the ghost nodes.  Element
 * values (owned and ghost elements) are averaged to the nodes, weighted by
 * element size.  Call after
 * goma_setup and before goma_solve or each goma_advance_*; Goma uses the
 * values from then on.  goma_solve and goma_advance_* return an error while
 * an imported field has no values.
 */
int goma_set_external_field(goma_context *ctx,
                            const char *name,
                            goma_field_location location,
                            const double *values);

/* End a problem advanced with goma_advance_*: free the solver and finish
 * the output files.  Called by goma_destroy if needed. */
int goma_finish(goma_context *ctx);

/*
 * Mesh of this rank, valid from goma_setup until goma_destroy (a problem
 * with mesh adaptation changes it while advancing).  Nodes are numbered
 * locally from 0, the nodes this rank owns first, then the ghost nodes owned
 * by other ranks.  Elements are numbered locally from 0, block by block; in
 * parallel, elements along a processor boundary are on both ranks and owned
 * by one of them.
 * Global ids are the 1-based node and element numbers of the undecomposed
 * mesh.  Coordinates are those of the mesh as read, without mesh
 * displacements.
 */
typedef struct goma_mesh_info {
  int dim;             /* spatial dimension */
  int num_nodes;       /* owned and ghost nodes */
  int num_owned_nodes; /* nodes 0 .. num_owned_nodes-1 are owned */
  int num_elems;       /* owned and ghost elements */
  int num_owned_elems;
  int num_elem_blocks;
} goma_mesh_info;

int goma_mesh_get_info(goma_context *ctx, goma_mesh_info *info);

/* xyz[num_nodes * dim], x, y (, z) of each node */
int goma_mesh_get_coordinates(goma_context *ctx, double *xyz);

/* ids[num_nodes] / ids[num_elems] */
int goma_mesh_get_node_global_ids(goma_context *ctx, long long *ids);
int goma_mesh_get_elem_global_ids(goma_context *ctx, long long *ids);

/* owners[num_elems]: rank (in the communicator given to goma_create) that
 * owns each element */
int goma_mesh_get_elem_owners(goma_context *ctx, int *owners);

/* Element block block (0 .. num_elem_blocks-1): its id in the mesh file, its
 * number of elements and nodes per element, and its connectivity
 * (*num_elems * *nodes_per_elem local node numbers, owned by Goma).  Any
 * output pointer may be NULL. */
int goma_mesh_get_block(goma_context *ctx,
                        int block,
                        int *block_id,
                        int *num_elems,
                        int *nodes_per_elem,
                        const int **connectivity);

/*
 * Free the context and memory allocated for the problem and reset global
 * state, sets *ctx to NULL.  May be called after any stage, including a
 * failed one.
 */
int goma_destroy(goma_context **ctx);

#ifdef __cplusplus
}
#endif

#endif /* GOMA_H */
