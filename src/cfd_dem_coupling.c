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
 * Goma <-> LAMMPS CFD-DEM coupler.
 *
 * Both codes run on MPI_COMM_WORLD and take turns.  The first version gathers every
 * particle onto every rank (lammps_gather_concat), which is simple and robust for
 * moderate particle counts; replace with a spatially distributed exchange if needed.
 *
 * Parallel bookkeeping relies on Goma's ghost layer: every element touching an owned node
 * is present locally, so depositing every particle into every local element it lies in
 * gives *complete* sums on owned nodes.  Owned values are then pushed to external nodes
 * with exchange_node().  A particle's drag is computed once, by the rank owning the
 * element that contains it (dpi->elem_owner), and shared with MPI_Allreduce.
 *
 * Deposition (eps and reaction force) uses element vertex (linear) shape functions and
 * lumped vertex volumes; see cfd_dem_util.h for why.  Values at higher-order nodes
 * (midside, face, center) are interpolated from the vertices.
 */

#include "cfd_dem_coupling.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mm_eh.h"
#include "std.h"

/* Input specs live here so they exist with or without LAMMPS. */
static struct Cfd_Dem_Specs cfd_dem_storage = {
    .enabled = FALSE,
    .lammps_input = "",
    .dem_max_dt = -1.0,
    .drag_model = CFD_DEM_DRAG_DI_FELICE,
    .coupling_mode = CFD_DEM_COUPLE_TWO_WAY,
    .fix_id = "cfd_drag",
    .eps_min = 0.35,
    .depth_2d = -1.0,
    .drag_field_names = {"DEM_FX", "DEM_FY", "DEM_FZ"},
    .drag_field_index = {-1, -1, -1},
};
struct Cfd_Dem_Specs *cfd_dem = &cfd_dem_storage;

#ifdef GOMA_ENABLE_LAMMPS

#include <mpi.h>

#include "cfd_dem_util.h"
#include "density.h"
#include "dp_comm.h"
#include "el_elm.h"
#include "el_elm_info.h"
#include "el_geom.h"
#include "load_field_variables.h"
#include "mm_as.h"
#include "mm_as_alloc.h"
#include "mm_as_structs.h"
#include "mm_fill_ptrs.h"
#include "mm_fill_util.h"
#include "mm_mp.h"
#include "mm_mp_structs.h"
#include "mm_unknown_map.h"
#include "mm_viscosity.h"
#include "rd_mesh.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_io.h"
#include "rf_mp.h"
#include "rf_solve.h"

#include "lammps/library.h"

/* ------------------------------------------------------------------------- */
/* coupler state                                                             */
/* ------------------------------------------------------------------------- */

typedef struct {
  void *lmp;       /* LAMMPS handle                                     */
  int lmp_dim;     /* LAMMPS dimension (2 or 3)                         */
  int dim;         /* goma mesh dimension                               */
  int first_run;   /* next LAMMPS run needs "pre yes"                   */
  double last_dt;  /* last DEM timestep set                             */
  double depth;    /* out-of-plane depth multiplying 2D mesh volumes     */
  Exo_DB *exo;
  Dpi *dpi;

  /* mesh: per local element */
  int num_elems;
  int *shape;          /* element shape (TRIANGLE, QUADRILATERAL, ...)  */
  int *nverts;         /* vertices per element                          */
  double (*bbox)[6];   /* lo[3], hi[3]                                   */
  int *hi_ptr;         /* CSR into hi_node/hi_wt: higher-order nodes    */
  int *hi_node;        /* local node id                                 */
  double *hi_wt;       /* CFD_DEM_MAX_VERTS weights per entry           */

  /* bin grid over the local bounding box for point location */
  double grid_lo[3], grid_h[3];
  int grid_n[3];
  int *bin_ptr, *bin_elem;

  /* nodal arrays (local nodes) */
  int num_nodes;
  int *is_vertex;      /* 1 for element vertices                        */
  double *node_vol;    /* lumped vertex volume                          */
  double *node_eps;    /* fluid fraction                                */
  double *node_f[DIM]; /* fluid reaction force per volume               */
  int eps_valid;

  /* particles (every particle known on every rank) */
  int np, np_cap;
  int *pid;
  double *px, *pv, *prad; /* 3*np, 3*np, np */
  int *pelem;             /* local element or -1                     */
  double *pxi;            /* 3*np parametric coordinates             */
  int *powner;            /* owning rank, Num_Proc if outside mesh    */
  double *beta;           /* np drag coefficient, F_d = beta (u - v)  */
  double *ufl;            /* 3*np fluid velocity at the particle     */
  double *fextra;         /* 3*np non-drag fluid force (pressure)    */
  double *drag_acc;       /* 3*np drag summed over DEM sub-steps     */
  int64_t run_start_step; /* LAMMPS step at the start of the run    */
  int id_max;
  int *id_to_index; /* [id_max+1] */
} cfd_dem_state;

static cfd_dem_state S;

static void lmp_check(const char *what) {
  if (lammps_has_error(S.lmp)) {
    char buf[1024];
    lammps_get_last_error_message(S.lmp, buf, sizeof(buf));
    GOMA_EH(GOMA_ERROR, "LAMMPS error during %s: %s", what, buf);
  }
}

static void lmp_cmd(const char *cmd) {
  lammps_command(S.lmp, cmd);
  lmp_check(cmd);
}

static int elem_shape(int ielem) { return type2shape(Elem_Type(S.exo, ielem)); }

static void elem_vertex_coords(int ielem, double coords[][3]) {
  const int nv = S.nverts[ielem];
  const int base = S.exo->elem_ptr[ielem];
  for (int i = 0; i < nv; i++) {
    const int node = S.exo->node_list[base + i];
    for (int k = 0; k < 3; k++)
      coords[i][k] = (k < S.dim) ? Coor[k][node] : 0.0;
  }
}

/* ------------------------------------------------------------------------- */
/* mesh setup                                                                */
/* ------------------------------------------------------------------------- */

static void setup_mesh(void) {
  Exo_DB *exo = S.exo;
  const int ne = exo->num_elems;
  int e, i, k;

  S.num_elems = ne;
  S.num_nodes = exo->num_nodes;
  S.shape = calloc(ne, sizeof(int));
  S.nverts = calloc(ne, sizeof(int));
  S.bbox = calloc(ne, sizeof(*S.bbox));
  S.hi_ptr = calloc(ne + 1, sizeof(int));
  S.is_vertex = calloc(S.num_nodes, sizeof(int));
  S.node_vol = calloc(S.num_nodes, sizeof(double));
  S.node_eps = calloc(S.num_nodes, sizeof(double));
  for (k = 0; k < DIM; k++)
    S.node_f[k] = calloc(S.num_nodes, sizeof(double));

  /* shapes, vertices, bounding boxes, lumped vertex volumes */
  int n_hi = 0;
  for (e = 0; e < ne; e++) {
    S.shape[e] = elem_shape(e);
    S.nverts[e] = cfd_dem_num_vertices(S.shape[e]);
    if (S.nverts[e] < 0 || cfd_dem_shape_dim(S.shape[e]) != S.dim) {
      GOMA_EH(GOMA_ERROR,
              "CFD-DEM coupling supports only tri/quad (2D) and tet/hex (3D) bulk elements");
    }
    const int nn = elem_info(NNODES, Elem_Type(exo, e));
    n_hi += nn - S.nverts[e];

    double coords[CFD_DEM_MAX_VERTS][3], vol[CFD_DEM_MAX_VERTS];
    elem_vertex_coords(e, coords);
    for (k = 0; k < 3; k++) {
      S.bbox[e][k] = 1e300;
      S.bbox[e][3 + k] = -1e300;
    }
    for (i = 0; i < S.nverts[e]; i++)
      for (k = 0; k < 3; k++) {
        S.bbox[e][k] = fmin(S.bbox[e][k], coords[i][k]);
        S.bbox[e][3 + k] = fmax(S.bbox[e][3 + k], coords[i][k]);
      }
    cfd_dem_vertex_volumes(S.shape[e], coords, vol);
    const int base = exo->elem_ptr[e];
    for (i = 0; i < S.nverts[e]; i++) {
      const int node = exo->node_list[base + i];
      S.is_vertex[node] = 1;
      S.node_vol[node] += vol[i] * S.depth;
    }
  }

  /* interpolation weights of higher-order nodes from the vertices (fixed mesh) */
  S.hi_node = calloc(n_hi > 0 ? n_hi : 1, sizeof(int));
  S.hi_wt = calloc((n_hi > 0 ? n_hi : 1) * CFD_DEM_MAX_VERTS, sizeof(double));
  int pos = 0;
  for (e = 0; e < ne; e++) {
    S.hi_ptr[e] = pos;
    const int nn = elem_info(NNODES, Elem_Type(exo, e));
    const int base = exo->elem_ptr[e];
    double coords[CFD_DEM_MAX_VERTS][3];
    elem_vertex_coords(e, coords);
    for (i = S.nverts[e]; i < nn; i++) {
      const int node = exo->node_list[base + i];
      double xn[3] = {0, 0, 0}, xi[3], N[CFD_DEM_MAX_VERTS];
      for (k = 0; k < S.dim; k++)
        xn[k] = Coor[k][node];
      cfd_dem_shape_center(S.shape[e], xi);
      if (cfd_dem_invert_map(S.shape[e], coords, xn, xi) != 0) {
        GOMA_EH(GOMA_ERROR, "CFD-DEM: could not invert vertex map for node %d of element %d",
                node, e);
      }
      cfd_dem_vertex_shape(S.shape[e], xi, N, NULL);
      S.hi_node[pos] = node;
      for (int v = 0; v < CFD_DEM_MAX_VERTS; v++)
        S.hi_wt[pos * CFD_DEM_MAX_VERTS + v] = (v < S.nverts[e]) ? N[v] : 0.0;
      pos++;
    }
  }
  S.hi_ptr[ne] = pos;

  /* bin grid: about one element per bin */
  double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
  for (e = 0; e < ne; e++)
    for (k = 0; k < 3; k++) {
      lo[k] = fmin(lo[k], S.bbox[e][k]);
      hi[k] = fmax(hi[k], S.bbox[e][3 + k]);
    }
  const double nb_target = fmax(1.0, (double)ne);
  double span[3], vol = 1.0;
  for (k = 0; k < S.dim; k++) {
    span[k] = fmax(hi[k] - lo[k], 1e-300);
    vol *= span[k];
  }
  const double h = pow(vol / nb_target, 1.0 / S.dim);
  for (k = 0; k < 3; k++) {
    if (k < S.dim && ne > 0) {
      S.grid_n[k] = (int)fmax(1.0, fmin(512.0, ceil(span[k] / h)));
      S.grid_lo[k] = lo[k];
      S.grid_h[k] = span[k] / S.grid_n[k];
    } else {
      S.grid_n[k] = 1;
      S.grid_lo[k] = 0.0;
      S.grid_h[k] = 1.0;
    }
  }
  const int nbins = S.grid_n[0] * S.grid_n[1] * S.grid_n[2];
  S.bin_ptr = calloc(nbins + 1, sizeof(int));
  int pass;
  for (pass = 0; pass < 2; pass++) {
    int *fill = (pass == 1) ? calloc(nbins, sizeof(int)) : NULL;
    for (e = 0; e < ne; e++) {
      int b0[3], b1[3];
      for (k = 0; k < 3; k++) {
        if (k < S.dim) {
          b0[k] = (int)floor((S.bbox[e][k] - S.grid_lo[k]) / S.grid_h[k]);
          b1[k] = (int)floor((S.bbox[e][3 + k] - S.grid_lo[k]) / S.grid_h[k]);
          b0[k] = MAX(0, MIN(S.grid_n[k] - 1, b0[k]));
          b1[k] = MAX(0, MIN(S.grid_n[k] - 1, b1[k]));
        } else {
          b0[k] = b1[k] = 0;
        }
      }
      for (int a = b0[0]; a <= b1[0]; a++)
        for (int b = b0[1]; b <= b1[1]; b++)
          for (int c = b0[2]; c <= b1[2]; c++) {
            const int bin = (c * S.grid_n[1] + b) * S.grid_n[0] + a;
            if (pass == 0)
              S.bin_ptr[bin + 1]++;
            else
              S.bin_elem[S.bin_ptr[bin] + fill[bin]++] = e;
          }
    }
    if (pass == 0) {
      for (int b = 0; b < nbins; b++)
        S.bin_ptr[b + 1] += S.bin_ptr[b];
      S.bin_elem = calloc(S.bin_ptr[nbins] > 0 ? S.bin_ptr[nbins] : 1, sizeof(int));
    } else {
      free(fill);
    }
  }
}

/* Locate x in the local mesh; returns local element (or -1) and xi. */
static int locate_point(const double x[3], double xi[3]) {
  int b[3], k;
  for (k = 0; k < 3; k++) {
    if (k < S.dim) {
      b[k] = (int)floor((x[k] - S.grid_lo[k]) / S.grid_h[k]);
      if (b[k] < 0 || b[k] >= S.grid_n[k])
        return -1;
    } else {
      b[k] = 0;
    }
  }
  const int bin = (b[2] * S.grid_n[1] + b[1]) * S.grid_n[0] + b[0];
  const double tol = 1.0e-10;
  for (int j = S.bin_ptr[bin]; j < S.bin_ptr[bin + 1]; j++) {
    const int e = S.bin_elem[j];
    int inbox = 1;
    for (k = 0; k < S.dim; k++) {
      const double pad = tol * (S.bbox[e][3 + k] - S.bbox[e][k]);
      if (x[k] < S.bbox[e][k] - pad || x[k] > S.bbox[e][3 + k] + pad)
        inbox = 0;
    }
    if (!inbox)
      continue;
    double coords[CFD_DEM_MAX_VERTS][3];
    elem_vertex_coords(e, coords);
    cfd_dem_shape_center(S.shape[e], xi);
    if (cfd_dem_invert_map(S.shape[e], coords, x, xi) == 0 &&
        cfd_dem_xi_inside(S.shape[e], xi, 1.0e-8)) {
      return e;
    }
  }
  return -1;
}

/* ------------------------------------------------------------------------- */
/* particles                                                                 */
/* ------------------------------------------------------------------------- */

static void ensure_particle_capacity(int np) {
  if (np <= S.np_cap)
    return;
  S.np_cap = MAX(np, 2 * S.np_cap);
  S.pid = realloc(S.pid, S.np_cap * sizeof(int));
  S.px = realloc(S.px, 3 * S.np_cap * sizeof(double));
  S.pv = realloc(S.pv, 3 * S.np_cap * sizeof(double));
  S.prad = realloc(S.prad, S.np_cap * sizeof(double));
  S.pelem = realloc(S.pelem, S.np_cap * sizeof(int));
  S.pxi = realloc(S.pxi, 3 * S.np_cap * sizeof(double));
  S.powner = realloc(S.powner, S.np_cap * sizeof(int));
  S.beta = realloc(S.beta, S.np_cap * sizeof(double));
  S.ufl = realloc(S.ufl, 3 * S.np_cap * sizeof(double));
  S.fextra = realloc(S.fextra, 3 * S.np_cap * sizeof(double));
  S.drag_acc = realloc(S.drag_acc, 3 * S.np_cap * sizeof(double));
}

static double particle_volume(int p) {
  const double r = S.prad[p];
  return (S.lmp_dim == 3) ? (4.0 / 3.0) * M_PI * r * r * r : M_PI * r * r;
}

/* Gather all particles everywhere, locate them, and decide owners. */
static void gather_particles(void) {
  const double natoms = lammps_get_natoms(S.lmp);
  if (natoms > 2.0e9)
    GOMA_EH(GOMA_ERROR, "CFD-DEM: too many particles for the gather-all coupler");
  S.np = (int)natoms;
  ensure_particle_capacity(MAX(S.np, 1));

  if (S.np > 0) {
    lammps_gather_concat(S.lmp, "id", 0, 1, S.pid);
    lammps_gather_concat(S.lmp, "x", 1, 3, S.px);
    lammps_gather_concat(S.lmp, "v", 1, 3, S.pv);
    lammps_gather_concat(S.lmp, "radius", 1, 1, S.prad);
    lmp_check("gather particles (atom_style sphere is required)");
  }

  /* id -> index map, needed to fill fix external on each rank's local atoms */
  int id_max = 0;
  for (int p = 0; p < S.np; p++)
    id_max = MAX(id_max, S.pid[p]);
  if (id_max > S.id_max || S.id_to_index == NULL) {
    S.id_max = id_max;
    S.id_to_index = realloc(S.id_to_index, (id_max + 1) * sizeof(int));
  }
  for (int i = 0; i <= S.id_max; i++)
    S.id_to_index[i] = -1;
  for (int p = 0; p < S.np; p++)
    S.id_to_index[S.pid[p]] = p;

  for (int p = 0; p < S.np; p++) {
    double xp[3] = {S.px[3 * p], S.px[3 * p + 1], S.dim == 3 ? S.px[3 * p + 2] : 0.0};
    S.pelem[p] = locate_point(xp, &S.pxi[3 * p]);
    S.powner[p] = (S.pelem[p] >= 0 && S.dpi->elem_owner[S.pelem[p]] == ProcID) ? ProcID : Num_Proc;
  }
  if (S.np > 0)
    MPI_Allreduce(MPI_IN_PLACE, S.powner, S.np, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  int outside = 0;
  for (int p = 0; p < S.np; p++)
    if (S.powner[p] == Num_Proc)
      outside++;
  if (outside > 0)
    P0PRINTF("CFD-DEM: %d of %d particles lie outside the fluid mesh (no coupling)\n", outside,
             S.np);
}

/* ------------------------------------------------------------------------- */
/* deposition: particle quantities -> nodal fields                           */
/* ------------------------------------------------------------------------- */

/* Push owned vertex values to external nodes, then fill higher-order nodes. */
static void finish_nodal_field(double *f) {
  exchange_node(cx[0], S.dpi, f);
  for (int e = 0; e < S.num_elems; e++) {
    const int base = S.exo->elem_ptr[e];
    for (int j = S.hi_ptr[e]; j < S.hi_ptr[e + 1]; j++) {
      double val = 0.0;
      for (int v = 0; v < S.nverts[e]; v++)
        val += S.hi_wt[j * CFD_DEM_MAX_VERTS + v] * f[S.exo->node_list[base + v]];
      f[S.hi_node[j]] = val;
    }
  }
}

static void compute_void_fraction(void) {
  double *solid = calloc(S.num_nodes, sizeof(double));
  for (int p = 0; p < S.np; p++) {
    const int e = S.pelem[p];
    if (e < 0)
      continue;
    double N[CFD_DEM_MAX_VERTS];
    cfd_dem_vertex_shape(S.shape[e], &S.pxi[3 * p], N, NULL);
    const double Vp = particle_volume(p);
    const int base = S.exo->elem_ptr[e];
    for (int v = 0; v < S.nverts[e]; v++)
      solid[S.exo->node_list[base + v]] += Vp * N[v];
  }
  int clipped = 0;
  for (int i = 0; i < S.num_nodes; i++) {
    S.node_eps[i] = 1.0;
    if (S.is_vertex[i] && S.node_vol[i] > 0.0) {
      double eps = 1.0 - solid[i] / S.node_vol[i];
      if (eps < cfd_dem->eps_min) {
        if (i < S.dpi->num_owned_nodes)
          clipped++;
        eps = cfd_dem->eps_min;
      }
      S.node_eps[i] = eps;
    }
  }
  free(solid);
  finish_nodal_field(S.node_eps);
  S.eps_valid = TRUE;

  MPI_Allreduce(MPI_IN_PLACE, &clipped, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (clipped > 0)
    P0PRINTF("CFD-DEM: void fraction clipped to %g at %d nodes\n", cfd_dem->eps_min, clipped);
}

/* Fluid reaction force per unit volume: -sum_p w_i F_fluid,p / V_i */
static void compute_reaction_field(const double *fluid_force) {
  for (int k = 0; k < DIM; k++)
    memset(S.node_f[k], 0, S.num_nodes * sizeof(double));
  if (cfd_dem->coupling_mode == CFD_DEM_COUPLE_TWO_WAY) {
    for (int p = 0; p < S.np; p++) {
      const int e = S.pelem[p];
      if (e < 0)
        continue;
      double N[CFD_DEM_MAX_VERTS];
      cfd_dem_vertex_shape(S.shape[e], &S.pxi[3 * p], N, NULL);
      const int base = S.exo->elem_ptr[e];
      for (int v = 0; v < S.nverts[e]; v++)
        for (int k = 0; k < S.dim; k++)
          S.node_f[k][S.exo->node_list[base + v]] -= N[v] * fluid_force[3 * p + k];
    }
    for (int i = 0; i < S.num_nodes; i++)
      for (int k = 0; k < S.dim; k++)
        S.node_f[k][i] = (S.is_vertex[i] && S.node_vol[i] > 0.0) ? S.node_f[k][i] / S.node_vol[i]
                                                                  : 0.0;
  }
  for (int k = 0; k < S.dim; k++) {
    finish_nodal_field(S.node_f[k]);
    const int w = cfd_dem->drag_field_index[k];
    if (w >= 0)
      memcpy(efv->ext_fld_ndl_val[w], S.node_f[k], S.num_nodes * sizeof(double));
  }

  /* conservation diagnostic: total reaction on fluid vs. total fluid-side particle force */
  double sums[2 * DIM] = {0};
  for (int p = 0; p < S.np; p++)
    if (S.powner[p] == ProcID)
      for (int k = 0; k < S.dim; k++)
        sums[k] += fluid_force[3 * p + k];
  for (int i = 0; i < S.dpi->num_owned_nodes; i++)
    for (int k = 0; k < S.dim; k++)
      sums[DIM + k] += S.node_f[k][i] * S.node_vol[i];
  MPI_Allreduce(MPI_IN_PLACE, sums, 2 * DIM, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (cfd_dem->coupling_mode == CFD_DEM_COUPLE_TWO_WAY) {
    P0PRINTF("CFD-DEM: drag on particles = (%g, %g, %g), reaction on fluid = (%g, %g, %g)\n",
             sums[0], sums[1], S.dim == 3 ? sums[2] : 0.0, sums[DIM], sums[DIM + 1],
             S.dim == 3 ? sums[DIM + 2] : 0.0);
  }
}

/* ------------------------------------------------------------------------- */
/* drag                                                                      */
/* ------------------------------------------------------------------------- */

/* Load Goma field variables in element e at parametric xi using solution x. */
static void load_fields_at(
    int e, const double xi[3], double *x, double *x_old, double *xdot, double *xdot_old) {
  int err = load_elem_dofptr(e, S.exo, x, x_old, xdot, xdot_old, 0);
  GOMA_EH(err, "load_elem_dofptr");
  err = bf_mp_init(pd);
  GOMA_EH(err, "bf_mp_init");
  double xi_local[DIM] = {xi[0], xi[1], xi[2]};
  err = load_basis_functions(xi_local, bfd);
  GOMA_EH(err, "load_basis_functions");
  err = beer_belly();
  GOMA_EH(err, "beer_belly");
  err = load_fv();
  GOMA_EH(err, "load_fv");
  err = load_bf_grad();
  GOMA_EH(err, "load_bf_grad");
  err = load_fv_grads();
  GOMA_EH(err, "load_fv_grads");
}

/*
 * Particle-fluid coupling coefficients, frozen over one goma step.
 *
 * For each owned particle the converged fluid at t^{n+1} is sampled at the particle and
 * the drag is linearized as F_d = beta (u_f - v).  The LAMMPS callback then applies
 *     F_p = c_d beta (u_f - v(t)) + F_extra
 * on every DEM sub-step with the particle's *current* velocity, so particles relax on the
 * DEM time scale even when the goma step is much longer than the particle response time.
 *
 * Force partition follows Zhou, Kuang, Chu & Yu (2010), J. Fluid Mech. 661, 482-510:
 *   Model A (div(eps T) in the fluid): c_d = 1, F_extra = -V_p grad p, and the fluid reaction
 *            is the drag only (the pressure part is carried by the eps grad p term).
 *   Model B (div(T) in the fluid):     c_d = 1/eps, F_extra = 0, fluid reaction is the same.
 * Viscous-stress, lift and virtual-mass forces are not included.
 */
static void compute_coupling_coefficients(
    double *x, double *x_old, double *xdot, double *xdot_old, double time) {
  memset(S.beta, 0, S.np * sizeof(double));
  memset(S.ufl, 0, 3 * S.np * sizeof(double));
  memset(S.fextra, 0, 3 * S.np * sizeof(double));
  if (cfd_dem->coupling_mode == CFD_DEM_COUPLE_VOID_FRACTION)
    return;

  for (int p = 0; p < S.np; p++) {
    if (S.powner[p] != ProcID)
      continue;
    load_fields_at(S.pelem[p], &S.pxi[3 * p], x, x_old, xdot, xdot_old);

    const double rho = density(NULL, time);
    double gamma_dot[DIM][DIM];
    for (int a = 0; a < DIM; a++)
      for (int b = 0; b < DIM; b++)
        gamma_dot[a][b] = fv->grad_v[a][b] + fv->grad_v[b][a];
    const double mu = viscosity(gn, gamma_dot, NULL);
    const double eps = pd->v[pg->imtrx][VOID_FRACTION] ? fv->void_fraction : 1.0;

    double umag = 0.0;
    for (int k = 0; k < S.dim; k++) {
      S.ufl[3 * p + k] = fv->v[k];
      umag += (fv->v[k] - S.pv[3 * p + k]) * (fv->v[k] - S.pv[3 * p + k]);
    }
    umag = sqrt(umag);
    double beta =
        cfd_dem_drag_coefficient(cfd_dem->drag_model, rho, mu, eps, 2.0 * S.prad[p], umag);
    if (beta < 0.0)
      GOMA_EH(GOMA_ERROR, "unknown CFD-DEM drag model");

    if (upd->vans_model == 2) {
      beta /= eps;
    } else {
      const double Vp = particle_volume(p);
      for (int k = 0; k < S.dim; k++)
        S.fextra[3 * p + k] = -Vp * fv->grad_P[k];
    }
    S.beta[p] = beta;
  }
  if (S.np > 0) {
    MPI_Allreduce(MPI_IN_PLACE, S.beta, S.np, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, S.ufl, 3 * S.np, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, S.fextra, 3 * S.np, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
  if (getenv("CFD_DEM_DEBUG") != NULL) {
    for (int p = 0; p < MIN(S.np, 4); p++) {
      if (S.powner[p] == ProcID) {
        fprintf(stdout,
                "CFD-DEM debug [rank %d] p=%d id=%d elem=%d xi=(%g,%g) x=(%g,%g) v=(%g,%g) "
                "beta=%g ufl=(%g,%g) fextra=(%g,%g)\n",
                ProcID, p, S.pid[p], S.pelem[p], S.pxi[3 * p], S.pxi[3 * p + 1], S.px[3 * p],
                S.px[3 * p + 1], S.pv[3 * p], S.pv[3 * p + 1], S.beta[p], S.ufl[3 * p],
                S.ufl[3 * p + 1], S.fextra[3 * p], S.fextra[3 * p + 1]);
        fflush(stdout);
      }
    }
  }
}

/*
 * fix external pf/callback: called by LAMMPS every DEM step (post_force) for local atoms.
 * Also accumulates the drag actually applied, so the fluid receives the time-averaged
 * reaction (exact momentum exchange over the coupling interval).
 */
static void fix_external_callback(
    void *ptr, int64_t timestep, int nlocal, int *ids, double **xatom, double **fexternal) {
  (void)ptr, (void)timestep, (void)xatom;
  double **v = (double **)lammps_extract_atom(S.lmp, "v");
  for (int i = 0; i < nlocal; i++) {
    const int id = ids[i];
    const int p = (id >= 0 && id <= S.id_max) ? S.id_to_index[id] : -1;
    fexternal[i][0] = fexternal[i][1] = fexternal[i][2] = 0.0;
    if (p < 0 || S.beta[p] == 0.0)
      continue;
    for (int k = 0; k < S.dim; k++) {
      const double fd = S.beta[p] * (S.ufl[3 * p + k] - v[i][k]);
      fexternal[i][k] = fd + S.fextra[3 * p + k];
      /* setup() also calls post_force; only accumulate the run's own steps */
      if (timestep > S.run_start_step)
        S.drag_acc[3 * p + k] += fd;
    }
  }
}

/* ------------------------------------------------------------------------- */
/* public API                                                                */
/* ------------------------------------------------------------------------- */

int cfd_dem_init(Exo_DB *exo, Dpi *dpi, double *x, double time) {
  (void)x;
  memset(&S, 0, sizeof(S));
  S.exo = exo;
  S.dpi = dpi;
  S.dim = exo->num_dim;
  S.first_run = TRUE;

  if (sizeof(int) != (size_t)lammps_extract_setting(NULL, "tagint"))
    GOMA_EH(GOMA_ERROR, "CFD-DEM coupling requires LAMMPS built with LAMMPS_SIZES=smallbig");

  char *argv[] = {"goma", "-log", "log.lammps", "-screen", "none", NULL};
  S.lmp = lammps_open(5, argv, MPI_COMM_WORLD, NULL);
  if (S.lmp == NULL)
    GOMA_EH(GOMA_ERROR, "CFD-DEM: lammps_open failed");
  lammps_file(S.lmp, cfd_dem->lammps_input);
  lmp_check(cfd_dem->lammps_input);

  S.lmp_dim = lammps_extract_setting(S.lmp, "dimension");
  if (S.dim == 3 && S.lmp_dim != 3)
    GOMA_EH(GOMA_ERROR, "CFD-DEM: a 3D goma mesh needs a 3D LAMMPS simulation");
  S.depth = 1.0;
  if (S.dim == 2 && S.lmp_dim == 3) {
    if (cfd_dem->depth_2d <= 0.0)
      GOMA_EH(GOMA_ERROR, "CFD-DEM: 2D mesh with 3D particles requires \"DEM 2D Depth\"");
    S.depth = cfd_dem->depth_2d;
  }
  if (lammps_extract_atom(S.lmp, "radius") == NULL && lammps_get_natoms(S.lmp) > 0)
    GOMA_EH(GOMA_ERROR, "CFD-DEM: LAMMPS must use atom_style sphere (per-atom radius)");

  /* the fix external that carries fluid forces is owned by the coupler */
  if (lammps_has_id(S.lmp, "fix", cfd_dem->fix_id)) {
    GOMA_EH(GOMA_ERROR,
            "CFD-DEM: fix %s already exists in %s; remove it, goma creates "
            "\"fix %s all external pf/callback 1 1\" itself",
            cfd_dem->fix_id, cfd_dem->lammps_input, cfd_dem->fix_id);
  }
  {
    char fcmd[256];
    snprintf(fcmd, sizeof(fcmd), "fix %s all external pf/callback 1 1", cfd_dem->fix_id);
    lmp_cmd(fcmd);
    /* note: LAMMPS silently ignores the call if the user pointer is NULL */
    lammps_set_fix_external_callback(S.lmp, cfd_dem->fix_id, fix_external_callback, &S);
    lmp_check("lammps_set_fix_external_callback");
  }

  /* synchronize clocks */
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "reset_timestep 0 time %.17g", time);
  lmp_cmd(cmd);

  /* resolve drag field indices among the external fields */
  for (int k = 0; k < DIM; k++) {
    cfd_dem->drag_field_index[k] = -1;
    if (k >= S.dim)
      continue;
    for (int w = 0; w < efv->Num_external_field; w++)
      if (strcmp(efv->name[w], cfd_dem->drag_field_names[k]) == 0)
        cfd_dem->drag_field_index[k] = w;
    if (cfd_dem->coupling_mode == CFD_DEM_COUPLE_TWO_WAY && cfd_dem->drag_field_index[k] < 0) {
      GOMA_EH(GOMA_ERROR,
              "CFD-DEM two-way coupling needs \"External Field = %s Q1 CFD_DEM\" in the input",
              cfd_dem->drag_field_names[k]);
    }
    const int w = cfd_dem->drag_field_index[k];
    if (w >= 0 && efv->ext_fld_ndl_val[w] == NULL)
      efv->ext_fld_ndl_val[w] = calloc(exo->num_nodes, sizeof(double));
  }

  setup_mesh();
  gather_particles();
  compute_void_fraction();
  {
    double *zero = calloc(3 * MAX(S.np, 1), sizeof(double));
    compute_reaction_field(zero);
    free(zero);
  }

  P0PRINTF("CFD-DEM: LAMMPS initialized from %s, %d particles, LAMMPS dimension %d\n",
           cfd_dem->lammps_input, S.np, S.lmp_dim);
  return 0;
}

int cfd_dem_apply_void_fraction(
    double *x, double *x_old, double *xdot, double *xdot_old, double delta_t, double theta) {
  if (!S.eps_valid)
    return -1;
  int written = 0;
  for (int i = 0; i < S.num_nodes; i++) {
    for (int mn = 0; mn < upd->Num_Mat; mn++) {
      const int j = Index_Solution(i, VOID_FRACTION, 0, 0, mn, pg->imtrx);
      if (j < 0)
        continue;
      x[j] = S.node_eps[i];
      /* Newton leaves FIX'd dofs alone, so set xdot consistent with the time integrator */
      if (delta_t > 0.0)
        xdot[j] = (1.0 + 2.0 * theta) / delta_t * (x[j] - x_old[j]) - 2.0 * theta * xdot_old[j];
      written++;
    }
  }
  if (written == 0)
    GOMA_EH(GOMA_ERROR, "CFD-DEM: no VOID_FRACTION unknowns found (EQ = void_fraction ...)");
  return 0;
}

int cfd_dem_advance(double *x,
                    double *x_old,
                    double *xdot,
                    double *xdot_old,
                    double time_old,
                    double time_new,
                    Exo_DB *exo,
                    Dpi *dpi) {
  (void)exo;
  (void)dpi;
  const double dt = time_new - time_old;
  if (dt <= 0.0)
    return 0;

  /* 1. drag coefficients and fluid velocity at the particles from the fluid at t^{n+1} */
  compute_coupling_coefficients(x, x_old, xdot, xdot_old, time_new);
  memset(S.drag_acc, 0, 3 * S.np * sizeof(double));
  S.run_start_step = (int64_t)(*(int64_t *)lammps_extract_global(S.lmp, "ntimestep"));

  /* 2. advance LAMMPS from time_old to time_new */
  int nsub = 1;
  if (cfd_dem->dem_max_dt > 0.0)
    nsub = (int)ceil(dt / cfd_dem->dem_max_dt * (1.0 - 1.0e-12));
  nsub = MAX(nsub, 1);
  const double dem_dt = dt / nsub;
  char cmd[256];
  int pre = S.first_run;
  if (fabs(dem_dt - S.last_dt) > 1.0e-14 * dem_dt) {
    snprintf(cmd, sizeof(cmd), "timestep %.17g", dem_dt);
    lmp_cmd(cmd);
    S.last_dt = dem_dt;
    pre = TRUE;
  }
  snprintf(cmd, sizeof(cmd), "run %d pre %s post no", nsub, pre ? "yes" : "no");
  lmp_cmd(cmd);
  S.first_run = FALSE;

  const double atime = *(double *)lammps_extract_global(S.lmp, "atime");
  if (fabs(atime - time_new) > 1.0e-9 * fmax(1.0, fabs(time_new))) {
    GOMA_WH(GOMA_ERROR, "CFD-DEM: LAMMPS time %.12g differs from goma time %.12g", atime,
            time_new);
  }
  P0PRINTF("CFD-DEM: LAMMPS advanced %d steps of %g to t=%g\n", nsub, dem_dt, atime);

  /* time-averaged drag actually applied to each particle over the interval */
  if (S.np > 0)
    MPI_Allreduce(MPI_IN_PLACE, S.drag_acc, 3 * S.np, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const int nsteps = nsub;
  const int np_old = S.np;
  int *pid_old = malloc(MAX(np_old, 1) * sizeof(int));
  double *fluid_force_old = malloc(3 * MAX(np_old, 1) * sizeof(double));
  memcpy(pid_old, S.pid, np_old * sizeof(int));
  for (int q = 0; q < 3 * np_old; q++)
    fluid_force_old[q] = S.drag_acc[q] / nsteps;

  /* 3. new particle state -> eps and fluid reaction for the next goma step */
  gather_particles();
  double *fluid_force = calloc(3 * MAX(S.np, 1), sizeof(double));
  for (int q = 0; q < np_old; q++) {
    const int id = pid_old[q];
    const int p = (id <= S.id_max) ? S.id_to_index[id] : -1;
    if (p >= 0)
      for (int k = 0; k < 3; k++)
        fluid_force[3 * p + k] = fluid_force_old[3 * q + k];
  }
  compute_void_fraction();
  compute_reaction_field(fluid_force);

  free(pid_old);
  free(fluid_force_old);
  free(fluid_force);
  return 0;
}

void cfd_dem_finalize(void) {
  if (S.lmp != NULL) {
    lammps_close(S.lmp);
    S.lmp = NULL;
  }
  free(S.shape);
  free(S.nverts);
  free(S.bbox);
  free(S.hi_ptr);
  free(S.hi_node);
  free(S.hi_wt);
  free(S.bin_ptr);
  free(S.bin_elem);
  free(S.is_vertex);
  free(S.node_vol);
  free(S.node_eps);
  for (int k = 0; k < DIM; k++)
    free(S.node_f[k]);
  free(S.pid);
  free(S.px);
  free(S.pv);
  free(S.prad);
  free(S.pelem);
  free(S.pxi);
  free(S.powner);
  free(S.beta);
  free(S.ufl);
  free(S.fextra);
  free(S.drag_acc);
  free(S.id_to_index);
  memset(&S, 0, sizeof(S));
}

#else /* !GOMA_ENABLE_LAMMPS */

int cfd_dem_init(Exo_DB *exo, Dpi *dpi, double *x, double time) {
  (void)exo, (void)dpi, (void)x, (void)time;
  GOMA_EH(GOMA_ERROR, "CFD-DEM coupling requires Goma built with -DENABLE_LAMMPS=ON");
  return -1;
}
int cfd_dem_apply_void_fraction(
    double *x, double *x_old, double *xdot, double *xdot_old, double delta_t, double theta) {
  (void)x, (void)x_old, (void)xdot, (void)xdot_old, (void)delta_t, (void)theta;
  return -1;
}
int cfd_dem_advance(double *x,
                    double *x_old,
                    double *xdot,
                    double *xdot_old,
                    double time_old,
                    double time_new,
                    Exo_DB *exo,
                    Dpi *dpi) {
  (void)x, (void)x_old, (void)xdot, (void)xdot_old, (void)time_old, (void)time_new, (void)exo,
      (void)dpi;
  return -1;
}
void cfd_dem_finalize(void) {}

#endif /* GOMA_ENABLE_LAMMPS */
