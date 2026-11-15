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

#include <cstddef>
#ifdef GOMA_ENABLE_OMEGA_H
#include "adapt/omega_h_interface.h"

#include <Omega_h_adapt.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_class.hpp>
#include <Omega_h_config.h>
#include <Omega_h_defines.hpp>
#include <Omega_h_fail.hpp>
#include <Omega_h_few.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_macros.h>
#include <Omega_h_matrix.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_metric.hpp>
#include <Omega_h_profile.hpp>
#include <Omega_h_vector.hpp>
#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <set>
#include <string.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Omega_h_align.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_for.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_mark.hpp"

extern "C" {
#define DISABLE_CPP
#ifndef MAX_PDIM
#define MAX_PDIM 3 /* Maximum physical problem dimension    */
#endif
#include <mm_bc.h>

#include "dp_types.h"
#include "dpi.h"
#include "exo_conn.h"
#include "exo_struct.h"
#include "mm_as.h"
#include "mm_as_structs.h"
#include "mm_eh.h"
#include "mm_interface.h"
#include "rf_allo.h"
#include "rf_bc.h"
#include "rf_bc_const.h"
#include "rf_fem_const.h"
#include "rf_fill_const.h"
#include "rf_io.h"
#include "rf_io_const.h"
#include "rf_mp.h"
#include "std.h"
#include <exodusII.h>
#define IGNORE_CPP_DEFINE
#include "sl_util_structs.h"
#undef IGNORE_CPP_DEFINE
#include "adapt/resetup_problem.h"
#include "base_mesh.h"
#include "dp_ghost.h"
#include "el_elm.h"
#include "el_elm_info.h"
#include "mm_elem_block_structs.h"
#include "mm_mp.h"
#include "mm_unknown_map.h"
#include "rd_dpi.h"
#include "rd_exo.h"
#include "rd_mesh.h"
#include "rf_fem.h"
#include "rf_node_const.h"
#include "rf_solver.h"
#include "util/goma_normal.h"
#include "wr_dpi.h"
#include "wr_exo.h"

extern int ***Local_Offset;
extern int ***Dolphin;
extern int *NumUnknowns;    /* Number of unknown variables updated by this   */
extern int *NumExtUnknowns; /* Number of unknown variables updated by this   */
extern int ***idv;
extern int **local_ROT_list;
extern int rotation_allocated;
extern Comm_Ex **cx;
#include "bc/rotate_coordinates.h"

#undef DISABLE_CPP
}

// #define DEBUG_OMEGA_H

static bool power_of_two(int num) { return (num & (num - 1)) == 0; }

namespace goma {

using namespace Omega_h;
#define CALL(f)                                                                          \
  do {                                                                                   \
    auto f_err = (f);                                                                    \
    if (f_err != 0) {                                                                    \
      const char *errmsg;                                                                \
      const char *errfunc;                                                               \
      int errnum;                                                                        \
      ex_get_err(&errmsg, &errfunc, &errnum);                                            \
      Omega_h_fail("Exodus call %s failed (%d): %s: %s\n", #f, errnum, errfunc, errmsg); \
    }                                                                                    \
  } while (0)

namespace exodus {
#ifndef OMEGA_H_USE_SEACASEXODUS
enum ClassifyWith {
  NODE_SETS = 0x1,
  SIDE_SETS = 0x2,
};
#endif
static OMEGA_H_INLINE int side_osh2exo(int dim, int side) {
  switch (dim) {
  case 2:
    switch (side) {
    case 1:
      return 2;
    case 2:
      return 3;
    case 0:
      return 1;
    }
    return -1;
  case 3:
    switch (side) {
    case 1:
      return 1;
    case 2:
      return 2;
    case 3:
      return 3;
    case 0:
      return 4;
    }
    return -1;
  }
  return -1;
}

static void setup_names(int nnames, std::vector<char> &storage, std::vector<char *> &ptrs) {
  constexpr auto max_name_length = MAX_STR_LENGTH + 1;
  storage = std::vector<char>(std::size_t(nnames * max_name_length), '\0');
  ptrs = std::vector<char *>(std::size_t(nnames), nullptr);
  for (int i = 0; i < nnames; ++i) {
    ptrs[std::size_t(i)] = storage.data() + max_name_length * i;
  }
}

// Adapted mesh partition for this processor as it would appear in a Nemesis
// (ExodusII + parallel info) file: node/element numbers are 1-based.
struct AdaptedMesh {
  std::string title;
  int dim = 0;
  int num_nodes = 0;
  int num_elems = 0;
  std::vector<double> coords[3];

  int block_id = 0;
  std::string elem_type;
  int nodes_per_elem = 0;
  std::vector<int> conn;

  std::vector<int> ns_ids;
  std::vector<std::vector<int>> ns_nodes;
  std::vector<int> ss_ids;
  std::vector<std::vector<int>> ss_elems;
  std::vector<std::vector<int>> ss_sides; // ExodusII side numbers

  // Distributed processing information
  int num_nodes_global = 0;
  int num_elems_global = 0;
  std::vector<int> global_set_ids;
  std::vector<int> global_ns_node_counts;
  std::vector<int> global_ss_side_counts;
  std::vector<int> global_eb_ids;
  std::vector<int> global_eb_counts;
  std::vector<int> node_map; // global node ids
  std::vector<int> elem_map; // global element ids
  int num_internal_nodes = 0;
  int num_boundary_nodes = 0;
  int num_internal_elems = 0;
  std::vector<int> node_cmap_ids;
  std::vector<std::vector<int>> node_cmap_nodes;
  std::vector<std::vector<int>> node_cmap_procs;
};

static int *alloc_int_copy(const std::vector<int> &v) {
  int *a = alloc_int_1(static_cast<int>(v.size()), 0);
  std::copy(v.begin(), v.end(), a);
  return a;
}

// NULL for empty arrays, as rd_exo gets from smalloc(0). generate_ghost_elems
// relies on this for side sets with no sides on this processor.
static int *alloc_int_copy_or_null(const std::vector<int> &v) {
  return v.empty() ? NULL : alloc_int_copy(v);
}

static char *alloc_str_copy(const std::string &s, std::size_t len) {
  char *a = static_cast<char *>(calloc(std::max(len, s.size() + 1), sizeof(char)));
  memcpy(a, s.c_str(), s.size());
  return a;
}

// 0-based local nodes on each 1-based ExodusII side, matching ex_get_side_set_node_list
static std::vector<int> exodus_side_nodes(int dim, int side) {
  static const std::vector<std::vector<int>> tri3 = {{0, 1}, {1, 2}, {2, 0}};
  static const std::vector<std::vector<int>> tet4 = {{0, 1, 3}, {1, 2, 3}, {0, 3, 2}, {0, 2, 1}};
  const auto &table = (dim == 3) ? tet4 : tri3;
  if (side < 1 || side > static_cast<int>(table.size())) {
    GOMA_EH(GOMA_ERROR, "Omega_h conversion: bad side %d for dim %d", side, dim);
  }
  return table[side - 1];
}

// Fill exo the same way rd_exo(INIT | MESH | RES0) fills it from a file
// containing the adapted mesh (1-based numbering, zero_base not yet applied)
static void fill_exo(Exo_DB *x, const AdaptedMesh &m, const char *path) {
  x->state = EXODB_STATE_GRND;
  x->path = alloc_str_copy(path, 0);
  x->node_map_exists = FALSE;
  x->elem_map_exists = FALSE;
  x->elem_order_map_exists = (Linear_Solver == FRONT) ? TRUE : FALSE;
  x->elem_var_tab_exists = FALSE;
  x->mode = EX_READ;
  x->comp_wordsize = sizeof(dbl);
  x->io_wordsize = sizeof(dbl);
  x->exoid = -1;

  // INIT
  x->title = alloc_str_copy(m.title, MAX_SLENGTH);
  x->num_dim = m.dim;
  x->num_nodes = m.num_nodes;
  x->num_elems = m.num_elems;
  x->num_elem_blocks = 1;
  x->num_node_sets = static_cast<int>(m.ns_ids.size());
  x->num_side_sets = static_cast<int>(m.ss_ids.size());
  x->num_qa_rec = 0;
  x->num_info = 0;
  x->ns_node_len = 0;
  for (const auto &ns : m.ns_nodes) {
    x->ns_node_len += static_cast<int>(ns.size());
  }
  x->ns_distfact_len = 0;
  x->ss_elem_len = 0;
  for (const auto &ss : m.ss_elems) {
    x->ss_elem_len += static_cast<int>(ss.size());
  }
  x->ss_distfact_len = 0;
  x->ss_node_len = x->ss_elem_len * m.dim;
  x->eb_num_props = 1;
  x->ns_num_props = 1;
  x->ss_num_props = 1;
  x->num_times = 0;
  x->state |= EXODB_STATE_INIT;

  // MESH
  x->x_coord = NULL;
  x->y_coord = NULL;
  x->z_coord = NULL;
  dbl **coord_ptrs[3] = {&x->x_coord, &x->y_coord, &x->z_coord};
  for (int d = 0; d < m.dim; d++) {
    *coord_ptrs[d] = alloc_dbl_1(m.num_nodes, 0.0);
    std::copy(m.coords[d].begin(), m.coords[d].end(), *coord_ptrs[d]);
  }
  // The adapted mesh never carried coordinate names, keep them empty
  x->coord_names = static_cast<char **>(smalloc(m.dim * sizeof(char *)));
  for (int d = 0; d < m.dim; d++) {
    x->coord_names[d] = alloc_str_copy("", MAX_STR_LENGTH + 1);
  }
  if (x->elem_order_map_exists && m.num_elems > 0) {
    x->elem_order_map = alloc_int_1(m.num_elems, 0);
    std::iota(x->elem_order_map, x->elem_order_map + m.num_elems, 1);
  }
  x->elem_eb = alloc_int_1(m.num_elems, 0);

  Element_Blocks = alloc_struct_1(ELEM_BLK_STRUCT, x->num_elem_blocks);
  x->eb_id = alloc_int_1(1, m.block_id);
  x->eb_elem_type = static_cast<char **>(smalloc(sizeof(char *)));
  x->eb_elem_type[0] = alloc_str_copy(m.elem_type, MAX_STR_LENGTH);
  x->eb_num_elems = alloc_int_1(1, m.num_elems);
  x->eb_num_nodes_per_elem = alloc_int_1(1, m.nodes_per_elem);
  x->eb_num_attr = alloc_int_1(1, 0);
  x->eb_conn = static_cast<int **>(smalloc(sizeof(int *)));
  x->eb_conn[0] = (m.num_elems > 0) ? alloc_int_copy(m.conn) : NULL;
  x->eb_attr = static_cast<dbl **>(smalloc(sizeof(dbl *)));
  x->eb_attr[0] = NULL;
  x->eb_ptr = alloc_int_1(2, 0);
  x->eb_ptr[1] = m.num_elems;

  ELEM_BLK_STRUCT *eb_ptr = Element_Blocks;
  eb_ptr->Elem_Blk_Num = 0;
  eb_ptr->Elem_Blk_Id = m.block_id;
  eb_ptr->Elem_Type =
      (m.num_elems > 0) ? get_type(x->eb_elem_type[0], m.nodes_per_elem, 0) : NULL_ELEM_TYPE;
  eb_ptr->Num_Nodes_Per_Elem = m.nodes_per_elem;
  eb_ptr->Num_Attr_Per_Elem = 0;
  int mindex = map_mat_index(eb_ptr->Elem_Blk_Id);
  eb_ptr->MatlProp_ptr = (mindex < 0) ? NULL : mp_glob[mindex];
  eb_ptr->ElemStorage = NULL;
  eb_ptr->Num_Elems_In_Block = m.num_elems;
  eb_ptr->IP_total = (m.num_elems > 0) ? elem_info(NQUAD, eb_ptr->Elem_Type) : 0;

  // Node sets
  if (x->num_node_sets > 0) {
    x->ns_id = alloc_int_copy(m.ns_ids);
    x->ns_num_nodes = alloc_int_1(x->num_node_sets, 0);
    x->ns_num_distfacts = alloc_int_1(x->num_node_sets, 0);
    x->ns_node_index = alloc_int_1(x->num_node_sets, 0);
    x->ns_distfact_index = alloc_int_1(x->num_node_sets, 0);
    if (x->ns_node_len > 0) {
      x->ns_node_list = alloc_int_1(x->ns_node_len, 0);
    }
    int offset = 0;
    for (int i = 0; i < x->num_node_sets; i++) {
      x->ns_num_nodes[i] = static_cast<int>(m.ns_nodes[i].size());
      x->ns_node_index[i] = offset;
      std::copy(m.ns_nodes[i].begin(), m.ns_nodes[i].end(), x->ns_node_list + offset);
      offset += x->ns_num_nodes[i];
    }
  }

  // Side sets
  if (x->num_side_sets > 0) {
    x->ss_id = alloc_int_copy(m.ss_ids);
    x->ss_num_sides = alloc_int_1(x->num_side_sets, 0);
    x->ss_num_distfacts = alloc_int_1(x->num_side_sets, 0);
    x->ss_elem_index = alloc_int_1(x->num_side_sets, 0);
    x->ss_distfact_index = alloc_int_1(x->num_side_sets, 0);
    x->ss_node_list = static_cast<int **>(smalloc(x->num_side_sets * sizeof(int *)));
    x->ss_node_cnt_list = static_cast<int **>(smalloc(x->num_side_sets * sizeof(int *)));
    x->ss_node_side_index = static_cast<int **>(smalloc(x->num_side_sets * sizeof(int *)));
    if (x->ss_elem_len > 0) {
      x->ss_elem_list = alloc_int_1(x->ss_elem_len, 0);
      x->ss_side_list = alloc_int_1(x->ss_elem_len, 0);
    }
    x->ss_node_list_exists = TRUE;
    int offset = 0;
    for (int i = 0; i < x->num_side_sets; i++) {
      int nsides = static_cast<int>(m.ss_elems[i].size());
      x->ss_num_sides[i] = nsides;
      x->ss_elem_index[i] = offset;
      std::copy(m.ss_elems[i].begin(), m.ss_elems[i].end(), x->ss_elem_list + offset);
      std::copy(m.ss_sides[i].begin(), m.ss_sides[i].end(), x->ss_side_list + offset);
      offset += nsides;

      std::vector<int> side_nodes;
      std::vector<int> side_node_counts;
      for (int j = 0; j < nsides; j++) {
        int elem = m.ss_elems[i][j] - 1;
        auto local = exodus_side_nodes(m.dim, m.ss_sides[i][j]);
        side_node_counts.push_back(static_cast<int>(local.size()));
        for (auto ln : local) {
          side_nodes.push_back(m.conn[elem * m.nodes_per_elem + ln]);
        }
      }
      x->ss_node_cnt_list[i] = alloc_int_copy_or_null(side_node_counts);
      x->ss_node_list[i] = alloc_int_copy_or_null(side_nodes);
      x->ss_node_side_index[i] = alloc_int_1(nsides + 1, 0);
      for (int j = 0; j < nsides; j++) {
        x->ss_node_side_index[i][j + 1] = x->ss_node_side_index[i][j] + x->ss_node_cnt_list[i][j];
      }
    }
  }

  // Properties, ExodusII always reports the "ID" property
  auto setup_id_prop = [](int num, const int *ids, char ***names, int ***props) {
    *names = static_cast<char **>(smalloc(sizeof(char *)));
    (*names)[0] = alloc_str_copy("ID", MAX_STR_LENGTH);
    *props = static_cast<int **>(smalloc(sizeof(int *)));
    (*props)[0] = alloc_int_1(num, 0);
    if (num > 0) {
      std::copy(ids, ids + num, (*props)[0]);
    }
  };
  setup_id_prop(x->num_node_sets, x->ns_id, &x->ns_prop_name, &x->ns_prop);
  setup_id_prop(x->num_side_sets, x->ss_id, &x->ss_prop_name, &x->ss_prop);
  setup_id_prop(x->num_elem_blocks, x->eb_id, &x->eb_prop_name, &x->eb_prop);
  x->state |= EXODB_STATE_MESH;

  // RES0, the adapted mesh carries no results
  x->num_glob_vars = 0;
  x->num_elem_vars = 0;
  x->num_node_vars = 0;
  x->state |= EXODB_STATE_RES0;
}

// Fill dpi the same way rd_dpi() fills it from a Nemesis file containing the
// adapted mesh, see rd_dpi.c
static void fill_dpi_parallel(Exo_DB *exo, Dpi *d, const AdaptedMesh &m) {
  init_dpi_struct(d);

  d->num_nodes_global = m.num_nodes_global;
  d->num_elems_global = m.num_elems_global;
  d->num_elem_blocks_global = static_cast<int>(m.global_eb_ids.size());
  d->num_node_sets_global = static_cast<int>(m.global_set_ids.size());
  d->num_side_sets_global = static_cast<int>(m.global_set_ids.size());

  d->ns_id_global = alloc_int_copy(m.global_set_ids);
  d->num_ns_global_node_counts = alloc_int_copy(m.global_ns_node_counts);
  d->num_ns_global_df_counts = alloc_int_1(d->num_node_sets_global, 0);
  d->ss_id_global = alloc_int_copy(m.global_set_ids);
  d->num_ss_global_side_counts = alloc_int_copy(m.global_ss_side_counts);
  d->num_ss_global_df_counts = alloc_int_1(d->num_side_sets_global, 0);
  d->global_elem_block_ids = alloc_int_copy(m.global_eb_ids);
  d->global_elem_block_counts = alloc_int_copy(m.global_eb_counts);

  d->rank = ProcID;
  d->num_proc = Num_Proc;
  d->num_proc_in_file = 1;
  d->ftype = 'p';

  d->node_index_global = alloc_int_copy(m.node_map);
  d->elem_index_global = alloc_int_copy(m.elem_map);
  exo->base_mesh->node_map = alloc_int_copy(m.node_map);
  exo->base_mesh->elem_map = alloc_int_copy(m.elem_map);

  d->num_internal_nodes = m.num_internal_nodes;
  d->num_boundary_nodes = m.num_boundary_nodes;
  d->num_external_nodes = 0;
  d->num_internal_elems = m.num_internal_elems;
  d->num_border_elems = 0;
  d->num_node_cmaps = static_cast<int>(m.node_cmap_ids.size());
  d->num_elem_cmaps = 0;

  d->base_internal_nodes = d->num_internal_nodes;
  d->base_boundary_nodes = d->num_boundary_nodes;
  d->base_external_nodes = d->num_external_nodes;
  d->base_internal_elems = d->num_internal_elems;
  d->base_border_elems = d->num_border_elems;

  d->proc_node_internal = alloc_int_1(d->num_internal_nodes, 0);
  std::iota(d->proc_node_internal, d->proc_node_internal + d->num_internal_nodes, 1);
  if (d->num_boundary_nodes > 0) {
    d->proc_node_boundary = alloc_int_1(d->num_boundary_nodes, 0);
    std::iota(d->proc_node_boundary, d->proc_node_boundary + d->num_boundary_nodes,
              d->num_internal_nodes + 1);
  }

  d->node_cmap_ids = alloc_int_copy(m.node_cmap_ids);
  d->node_cmap_node_counts = alloc_int_1(d->num_node_cmaps, 0);
  d->node_map_node_ids = static_cast<int **>(calloc(d->num_node_cmaps, sizeof(int *)));
  d->node_map_proc_ids = static_cast<int **>(calloc(d->num_node_cmaps, sizeof(int *)));
  for (int i = 0; i < d->num_node_cmaps; i++) {
    d->node_cmap_node_counts[i] = static_cast<int>(m.node_cmap_nodes[i].size());
    d->node_map_node_ids[i] = alloc_int_copy(m.node_cmap_nodes[i]);
    d->node_map_proc_ids[i] = alloc_int_copy(m.node_cmap_procs[i]);
  }

  // Derived information, as in rd_dpi()
  d->eb_id_global = static_cast<int *>(calloc(d->num_elem_blocks_global, sizeof(int)));
  std::vector<int> eb_num_nodes_local(d->num_elem_blocks_global, 0);
  d->eb_num_nodes_per_elem_global =
      static_cast<int *>(calloc(d->num_elem_blocks_global, sizeof(int)));
  for (int i = 0; i < exo->num_elem_blocks; i++) {
    for (int j = 0; j < d->num_elem_blocks_global; j++) {
      if (d->global_elem_block_ids[j] == exo->eb_id[i]) {
        eb_num_nodes_local[j] = exo->eb_num_nodes_per_elem[i];
        d->eb_id_global[i] = d->global_elem_block_ids[j];
      }
    }
  }

  d->ss_index_global = static_cast<int *>(malloc(sizeof(int) * exo->num_side_sets));
  for (int i = 0; i < exo->num_side_sets; i++) {
    for (int j = 0; j < d->num_side_sets_global; j++) {
      if (exo->ss_id[i] == d->ss_id_global[j]) {
        d->ss_index_global[i] = j;
      }
    }
  }

  std::vector<int> global_ss_internal(d->num_side_sets_global, 0);
  d->ss_internal_global = NULL;
  if (d->num_side_sets_global > 0) {
    int *local_ss_internal = find_ss_internal_boundary(exo);
    d->ss_internal_global = static_cast<int *>(calloc(d->num_side_sets_global, sizeof(int)));
    for (int i = 0; i < exo->num_side_sets; i++) {
      global_ss_internal[d->ss_index_global[i]] = local_ss_internal[i];
    }
    free(local_ss_internal);

    int neb = d->num_elem_blocks_global;
    int nss = d->num_side_sets_global;
    d->ss_block_index_global = static_cast<int *>(calloc(nss + 1, sizeof(int)));
    d->ss_block_list_global = static_cast<int *>(calloc(neb * nss, sizeof(int)));
    std::vector<int> ss_block_count_proc(neb * nss, 0);
    std::vector<int> ss_block_count_global(neb * nss, 0);

    for (int ss_id = 0; ss_id < exo->num_side_sets; ss_id++) {
      for (int elem_index = exo->ss_elem_index[ss_id];
           elem_index < (exo->ss_elem_index[ss_id] + exo->ss_num_sides[ss_id]); elem_index++) {
        int elem = exo->ss_elem_list[elem_index];
        int block = find_elemblock_index(elem, exo);
        if (block == -1) {
          GOMA_EH(GOMA_ERROR, "Element block not found ss_block_list");
        }
        for (int j = 0; j < neb; j++) {
          if (d->eb_id_global[j] == exo->eb_id[block]) {
            ss_block_count_proc[ss_id * neb + j] = 1;
          }
        }
      }
    }

    MPI_Allreduce(ss_block_count_proc.data(), ss_block_count_global.data(), neb * nss, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);

    int ss_block_index = 0;
    for (int ss_id = 0; ss_id < nss; ss_id++) {
      int ss_start = ss_block_index;
      for (int j = 0; j < neb; j++) {
        if (ss_block_count_global[ss_id * neb + j] > 0) {
          d->ss_block_list_global[ss_block_index] = j;
          ss_block_index++;
        }
      }
      d->ss_block_index_global[ss_id] = ss_start;
      d->ss_block_index_global[ss_id + 1] = ss_block_index;
    }
  }
  d->elem_owner = alloc_int_1(exo->num_elems, ProcID);

  MPI_Allreduce(eb_num_nodes_local.data(), d->eb_num_nodes_per_elem_global,
                d->num_elem_blocks_global, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(global_ss_internal.data(), d->ss_internal_global, d->num_side_sets_global, MPI_INT,
                MPI_MAX, MPI_COMM_WORLD);

  d->num_universe_nodes = d->num_internal_nodes + d->num_boundary_nodes + d->num_external_nodes;

  // zero_base maps
  for (int i = 0; i < exo->base_mesh->num_nodes; i++) {
    exo->base_mesh->node_map[i]--;
  }
  for (int i = 0; i < exo->base_mesh->num_elems; i++) {
    exo->base_mesh->elem_map[i]--;
  }
  zero_dpi(d);

  // setup node owners
  d->num_neighbors = d->num_node_cmaps;
  d->neighbor = alloc_int_1(d->num_neighbors, -1);
  d->node_owner = alloc_int_1(exo->num_nodes, ProcID);
  d->num_node_recv = alloc_int_1(d->num_neighbors, 0);
  d->num_node_send = alloc_int_1(d->num_neighbors, 0);
  for (int i = 0; i < d->num_node_cmaps; i++) {
    int neighbor = d->node_map_proc_ids[i][0];
    for (int j = 0; j < d->node_cmap_node_counts[i]; j++) {
      if (neighbor != d->node_map_proc_ids[i][j]) {
        GOMA_EH(GOMA_ERROR, "Unexpected proc id in node map proc ids");
      }
      if (neighbor < d->node_owner[d->node_map_node_ids[i][j]]) {
        d->node_owner[d->node_map_node_ids[i][j]] = neighbor;
        d->num_node_recv[i]++;
      }
    }
    d->neighbor[i] = neighbor;
  }

  int elem_offset = 0;
  for (int i = 0; i < exo->num_elem_blocks; i++) {
    int nnode_per_elem = exo->eb_num_nodes_per_elem[i];
    for (int j = 0; j < exo->eb_num_elems[i]; j++) {
      int min_proc = ProcID;
      for (int k = 0; k < nnode_per_elem; k++) {
        int proc = d->node_owner[exo->eb_conn[i][j * nnode_per_elem + k]];
        min_proc = std::min(min_proc, proc);
      }
      d->elem_owner[elem_offset] = min_proc;
      elem_offset++;
    }
  }

  goma_error err = generate_ghost_elems(exo, d);
  GOMA_EH(err, "generate_ghost_elements");

  // Exchange the global ids of the external (ghost) nodes with their owners and
  // verify the owner has them as border nodes
  int nn = d->num_neighbors;
  int offset = d->num_internal_nodes + d->num_boundary_nodes;
  std::vector<int> num_send_nodes(nn, 0);
  std::vector<int> num_recv_nodes(nn, 0);
  std::vector<std::vector<int>> global_send_nodes(nn);
  std::vector<std::vector<int>> global_recv_nodes(nn);
  std::vector<MPI_Request> requests(2 * nn);

  for (int i = 0; i < nn; i++) {
    MPI_Irecv(&num_send_nodes[i], 1, MPI_INT, d->neighbor[i], 206, MPI_COMM_WORLD,
              &requests[nn + i]);
  }
  for (int i = 0; i < nn; i++) {
    for (int j = 0; j < d->num_external_nodes; j++) {
      if (d->node_owner[offset + j] == d->neighbor[i]) {
        num_recv_nodes[i] += 1;
      }
    }
    MPI_Isend(&num_recv_nodes[i], 1, MPI_INT, d->neighbor[i], 206, MPI_COMM_WORLD, &requests[i]);
  }
  MPI_Waitall(2 * nn, requests.data(), MPI_STATUSES_IGNORE);

  for (int i = 0; i < nn; i++) {
    global_send_nodes[i].resize(num_send_nodes[i]);
    MPI_Irecv(global_send_nodes[i].data(), num_send_nodes[i], MPI_INT, d->neighbor[i], 207,
              MPI_COMM_WORLD, &requests[nn + i]);
  }
  for (int i = 0; i < nn; i++) {
    for (int j = 0; j < d->num_external_nodes; j++) {
      if (d->node_owner[offset + j] == d->neighbor[i]) {
        global_recv_nodes[i].push_back(d->node_index_global[offset + j]);
      }
    }
    MPI_Isend(global_recv_nodes[i].data(), num_recv_nodes[i], MPI_INT, d->neighbor[i], 207,
              MPI_COMM_WORLD, &requests[i]);
  }
  MPI_Waitall(2 * nn, requests.data(), MPI_STATUSES_IGNORE);

  for (int i = 0; i < nn; i++) {
    for (int j = 0; j < num_send_nodes[i]; j++) {
      if (in_list(global_send_nodes[i][j], d->num_internal_nodes, offset, d->node_index_global) ==
          -1) {
        GOMA_EH(GOMA_ERROR, "Required node to communicate doesn't exist in border nodes");
      }
    }
  }

  // reorder external nodes so they are grouped by neighbor
  int num_ext = d->num_external_nodes;
  std::vector<int> new_external_node_order(num_ext, 0);
  std::vector<int> old_to_new_external_node_order(num_ext, 0);
  std::vector<int> old_node_owner(d->node_owner + offset, d->node_owner + offset + num_ext);
  int new_index = 0;
  for (int j = 0; j < nn; j++) {
    for (int i = 0; i < num_ext; i++) {
      if (old_node_owner[i] == d->neighbor[j]) {
        new_external_node_order[new_index] = i;
        old_to_new_external_node_order[i] = new_index;
        new_index++;
      }
    }
  }
  if (new_index != num_ext) {
    GOMA_EH(GOMA_ERROR, "incorrect new ordering %d != %d", new_index, num_ext);
  }

  auto reorder_external = [&](auto *values) {
    std::vector<std::remove_pointer_t<decltype(values)>> old(values + offset,
                                                             values + offset + num_ext);
    for (int i = 0; i < num_ext; i++) {
      values[offset + i] = old[new_external_node_order[i]];
    }
  };
  reorder_external(d->node_index_global);
  reorder_external(exo->x_coord);
  if (exo->num_dim > 1) {
    reorder_external(exo->y_coord);
  }
  if (exo->num_dim > 2) {
    reorder_external(exo->z_coord);
  }

  auto renumber = [&](int &node) {
    if (node >= offset) {
      node = old_to_new_external_node_order[node - offset] + offset;
    }
  };
  for (int block = 0; block < exo->num_elem_blocks; block++) {
    int len = exo->eb_num_nodes_per_elem[block] * exo->eb_num_elems[block];
    for (int i = 0; i < len; i++) {
      renumber(exo->eb_conn[block][i]);
    }
  }
  for (int j = 0; j < exo->ns_node_len; j++) {
    renumber(exo->ns_node_list[j]);
  }
  for (int ins = 0; ins < exo->num_side_sets; ins++) {
    for (int lni = 0; lni < exo->ss_node_side_index[ins][exo->ss_num_sides[ins]]; lni++) {
      renumber(exo->ss_node_list[ins][lni]);
    }
  }
  for (int i = 0; i < num_ext; i++) {
    d->node_owner[offset + old_to_new_external_node_order[i]] = old_node_owner[i];
  }

  setup_ghost_to_base(exo, d);
  d->num_owned_nodes = d->num_internal_nodes + d->num_boundary_nodes;
  d->num_universe_nodes = d->num_internal_nodes + d->num_boundary_nodes + d->num_external_nodes;

  // The adapted mesh has no goma node set / side set consistency data
  d->goma_dpi_data = false;
  d->global_ns_node_len = 0;
  d->global_ss_elem_len = 0;
}

// In memory replacement for read_mesh_exoII() (rd_mesh.c) for the adapted mesh
static void setup_mesh_in_memory(Exo_DB *exo, Dpi *dpi, const AdaptedMesh &m, const char *path) {
  fill_exo(exo, m, path);
  zero_base(exo);
  goma_error error = setup_base_mesh(dpi, exo, Num_Proc);
  GOMA_EH(error, "setup_base_mesh");

  if (Num_Proc == 1) {
    uni_dpi(dpi, exo);
  } else {
    fill_dpi_parallel(exo, dpi, m);
  }

  SS_Internal_Boundary = alloc_int_1(exo->num_side_sets, INT_NOINIT);
  for (int ss_index = 0; ss_index < exo->num_side_sets; ss_index++) {
    int global_ss_index = dpi->ss_index_global[ss_index];
    SS_Internal_Boundary[ss_index] = dpi->ss_internal_global[global_ss_index];
  }

  setup_old_dpi(exo, dpi);
  setup_old_exo(exo, dpi, Num_Proc);

  int max_nodes =
      *std::max_element(dpi->eb_num_nodes_per_elem_global,
                        dpi->eb_num_nodes_per_elem_global + dpi->num_elem_blocks_global);
  if (max_nodes > MDE) {
    GOMA_EH(GOMA_ERROR, "The mesh has elements with %d nodes, MDE is %d", max_nodes, MDE);
  }

  check_sidesets(exo, BC_Types, Num_BC, dpi);
  check_nodesets(exo, BC_Types, Num_BC, dpi);
  check_elemblocks(exo, upd->Num_Mat, pd_glob, dpi);

  Matilda = alloc_int_1(exo->num_elem_blocks, 0);
  setup_matilda(exo, Matilda);

  build_elem_node(exo);
  build_node_elem(exo);
  build_elem_elem(exo);
  build_node_node(exo);
}

void convert_goma_to_omega_h(Exo_DB *exo, Dpi *dpi, double **x, Mesh *mesh, bool verbose) {
  std::vector<LO> local_to_global(exo->num_nodes);
  for (int i = 0; i < exo->num_nodes; i++) {
    local_to_global[i] = dpi->node_index_global[i];
  }

  std::vector<LO> owned_elems;
  for (int i = 0; i < exo->num_elems; i++) {
    if (dpi->elem_owner[i] == ProcID) {
      owned_elems.push_back(i);
    }
  }
  int nnodes_per_elem = exo->eb_num_nodes_per_elem[0];

  std::set<LO> local_verts;
  for (auto e : owned_elems) {
    for (int i = 0; i < nnodes_per_elem; i++) {
      local_verts.insert(exo->eb_conn[0][e * nnodes_per_elem + i]);
    }
  }

  std::vector<LO> old_locals(local_verts.begin(), local_verts.end());
  std::sort(old_locals.begin(), old_locals.end());
  std::vector<LO> new_local_verts(old_locals.size());
  std::iota(new_local_verts.begin(), new_local_verts.end(), 0);

  std::map<LO, LO> old_to_new;
  std::map<LO, LO> new_to_old;
  for (size_t i = 0; i < old_locals.size(); i++) {
    old_to_new.insert(std::pair<LO, LO>(old_locals[i], i));
    new_to_old.insert(std::pair<LO, LO>(i, old_locals[i]));
  }

  HostWrite<GO> vert_global;
  vert_global = decltype(vert_global)(GO(new_local_verts.size()), "global vertices");

  for (size_t i = 0; i < old_locals.size(); i++) {
    vert_global[i] = dpi->node_index_global[old_locals[i]];
  }

  HostWrite<LO> h_conn;
  h_conn = decltype(h_conn)(LO(owned_elems.size() * nnodes_per_elem), "host connectivity");

  for (size_t i = 0; i < owned_elems.size(); i++) {
    auto e = owned_elems[i];
    for (int j = 0; j < nnodes_per_elem; j++) {
      auto old_index = exo->eb_conn[0][e * nnodes_per_elem + j];
      auto new_index = old_to_new[old_index];
      h_conn[i * nnodes_per_elem + j] = new_index;
    }
  }

  auto dim = exo->num_dim;

  mesh->set_parting(OMEGA_H_ELEM_BASED);

  auto conn = LOs(h_conn.write());
  build_from_elems2verts(mesh, mesh->library()->world(), OMEGA_H_SIMPLEX, dim, conn,
                         GOs(vert_global));

  auto new_verts = mesh->globals(0);

  HostWrite<Real> h_coords(LO(new_verts.size() * dim));
  std::vector<LO> exo_from_omega;

  for (int i = 0; i < new_verts.size(); i++) {
    int idx = in_list(new_verts[i], 0, exo->num_nodes, dpi->node_index_global);
    assert(idx != -1);
    exo_from_omega.push_back(idx);

    double d_x = 0;
    double d_y = 0;
    double d_z = 0;

    // Get displacements to Anneal the mesh
    if (pd->gv[R_MESH1]) {
      int x_idx = Index_Solution(idx, R_MESH1, 0, 0, -1, pd->mi[R_MESH1]);
      int y_idx = Index_Solution(idx, R_MESH2, 0, 0, -1, pd->mi[R_MESH2]);
      int z_idx = -1;
      if (exo->num_dim == 3) {
        z_idx = Index_Solution(idx, R_MESH3, 0, 0, -1, pd->mi[R_MESH3]);
      }

      d_x = x[pd->mi[R_MESH1]][x_idx];
      d_y = x[pd->mi[R_MESH2]][y_idx];
      if (exo->num_dim == 3) {
        d_z = x[pd->mi[R_MESH3]][z_idx];
      }
    }

    h_coords[i * dim + 0] = exo->x_coord[idx] + d_x;
    h_coords[i * dim + 1] = exo->y_coord[idx] + d_y;
    if (exo->num_dim == 3) {
      h_coords[i * dim + 2] = exo->z_coord[idx] + d_z;
    }
  }

  auto coords = Reals(h_coords.write());
  mesh->add_coords(coords);
  Write<LO> elem_class_ids_w(LO(owned_elems.size()));
  for (size_t i = 0; i < owned_elems.size(); i++) {
    elem_class_ids_w[i] = 1;
  }
  std::map<LO, LO> exo_to_global;
  for (int node = 0; node < mesh->globals(0).size(); node++) {
    int exo_index = in_list(mesh->globals(0)[node], 0, exo->num_nodes, dpi->node_index_global);
    exo_to_global.insert(std::pair<LO, LO>(exo_index, node));
  }
  for (int j = V_FIRST; j < V_LAST; j++) {
    int imtrx = upd->matrix_index[j];

    if (imtrx >= 0) {
      if (j == MASS_FRACTION) {
        for (int mf = 0; mf < upd->Max_Num_Species; mf++) {
          auto var_values = Omega_h::Write<Omega_h::Real>(mesh->nverts());
          for (int i = 0; i < exo->num_nodes; i++) {
            if (exo_to_global.find(i) != exo_to_global.end()) {
              auto gnode = exo_to_global[i];
              int ja = Index_Solution(i, j, mf, 0, -2, imtrx);
              GOMA_EH(ja, "could not find solution");
              var_values[gnode] = x[imtrx][ja];
            }
          }
          std::string species_name = Exo_Var_Names[j].name2 + std::to_string(mf);
          mesh->add_tag(Omega_h::VERT, species_name, 1, Omega_h::Reals(var_values));
        }
      } else {
        auto var_values = Omega_h::Write<Omega_h::Real>(mesh->nverts());
        for (int i = 0; i < exo->num_nodes; i++) {
          if (exo_to_global.find(i) != exo_to_global.end()) {
            auto gnode = exo_to_global[i];
            int ja = Index_Solution(i, j, 0, 0, -2, imtrx);
            GOMA_EH(ja, "could not find solution");
            var_values[gnode] = x[imtrx][ja];
            if (tran->ale_adapt && (j >= R_MESH1 && j <= R_MESH3)) {
              var_values[gnode] = 0;
            }
          }
        }
        mesh->add_tag(Omega_h::VERT, Exo_Var_Names[j].name2, 1, Omega_h::Reals(var_values));

        if (tran->ale_adapt && j == R_MESH1 && ((ls == NULL) || (!ls->adapt))) {
          auto target_metrics =
              Omega_h::Write<Omega_h::Real>(mesh->nverts() * Omega_h::symm_ncomps(mesh->dim()));
          auto f0 = OMEGA_H_LAMBDA(Omega_h::LO index) {
            auto iso_size = tran->ale_adapt_iso_size;
            if (mesh->dim() == 2) {
              auto target_metric = Omega_h::compose_metric(Omega_h::identity_matrix<2, 2>(),
                                                           Omega_h::vector_2(iso_size, iso_size));
              Omega_h::set_vector(target_metrics, index, Omega_h::symm2vector(target_metric));
            } else {
              auto target_metric =
                  Omega_h::compose_metric(Omega_h::identity_matrix<3, 3>(),
                                          Omega_h::vector_3(iso_size, iso_size, iso_size));
              Omega_h::set_vector(target_metrics, index, Omega_h::symm2vector(target_metric));
            }
          };

          Omega_h::parallel_for(mesh->nverts(), f0, "set_iso_metric_values");
          mesh->add_tag(Omega_h::VERT, "iso_size_metric", Omega_h::symm_ncomps(mesh->dim()),
                        Omega_h::Reals(target_metrics));
        } else if (ls != NULL && ls->adapt && (j == FILL)) {
          auto target_metrics =
              Omega_h::Write<Omega_h::Real>(mesh->nverts() * Omega_h::symm_ncomps(mesh->dim()));
          auto f0 = OMEGA_H_LAMBDA(Omega_h::LO index) {
            auto F = var_values[index];
            auto iso_size = ls->adapt_outer_size;
            if (std::abs(F) < ls->adapt_width) {
              iso_size = ls->adapt_inner_size;
            }
            if (mesh->dim() == 2) {
              auto target_metric = Omega_h::compose_metric(Omega_h::identity_matrix<2, 2>(),
                                                           Omega_h::vector_2(iso_size, iso_size));
              Omega_h::set_vector(target_metrics, index, Omega_h::symm2vector(target_metric));
            } else {
              auto target_metric =
                  Omega_h::compose_metric(Omega_h::identity_matrix<3, 3>(),
                                          Omega_h::vector_3(iso_size, iso_size, iso_size));
              Omega_h::set_vector(target_metrics, index, Omega_h::symm2vector(target_metric));
            }
          };

          Omega_h::parallel_for(mesh->nverts(), f0, "set_iso_metric_values");
          mesh->add_tag(Omega_h::VERT, "iso_size_metric", Omega_h::symm_ncomps(mesh->dim()),
                        Omega_h::Reals(target_metrics));
        }
      }
    }
  }

  for (int w = 0; w < efv->Num_external_field; w++) {
    auto var_values = Omega_h::Write<Omega_h::Real>(mesh->nverts());
    for (int i = 0; i < exo->num_nodes; i++) {
      if (exo_to_global.find(i) != exo_to_global.end()) {
        auto gnode = exo_to_global[i];
        var_values[gnode] = efv->ext_fld_ndl_val[w][i];
      }
    }
    mesh->add_tag(Omega_h::VERT, efv->name[w], 1, Omega_h::Reals(var_values));
  }

  std::vector<int> side_set_ids(std::size_t(dpi->num_side_sets_global));
  for (int i = 0; i < dpi->num_side_sets_global; i++) {
    side_set_ids[i] = dpi->ss_id_global[i];
  }
  //  CALL(ex_get_ids(file, EX_SIDE_SET, side_set_ids.data()));
  Write<LO> side_class_ids_w(mesh->nents(dim - 1), -1);
  auto sides_are_exposed = mark_exposed_sides(mesh);
  classify_sides_by_exposure(mesh, sides_are_exposed);
  Write<I8> side_class_dims_w = deep_copy(mesh->get_array<I8>(dim - 1, "class_dim"));
  auto exposed_sides2side = collect_marked(sides_are_exposed);
  map_value_into(0, exposed_sides2side, side_class_ids_w);

  std::unordered_map<int, int> global_to_index;
  std::unordered_map<int, int> global_to_index_sides;
  for (int i = 0; i < mesh->nverts(); i++) {
    global_to_index[mesh->globals(0)[i]] = i;
  }
  for (int i = 0; i < mesh->globals(dim - 1).size(); i++) {
    global_to_index_sides[mesh->globals(dim - 1)[i]] = i;
  }

  if (1 && dpi->num_side_sets_global) {
    std::vector<char> names_memory;
    std::vector<char *> name_ptrs;
    setup_names(int(dpi->num_side_sets_global), names_memory, name_ptrs);
    //    CALL(ex_get_names(file, EX_SIDE_SET, name_ptrs.data()));
    for (int i = 0; i < dpi->num_side_sets_global; ++i) {
      assert(exo->ss_id[i] == dpi->ss_id_global[i]);
      int nsides;
      //      CALL(ex_get_set_param(file, EX_SIDE_SET, side_set_ids[i], &nsides, &ndist_factors));
      nsides = exo->ss_num_sides[i];
      if (verbose)
        std::cout << "Proc" << ProcID << " has " << nsides << "sides\n";
      int snl[MAX_NODES_PER_SIDE]; /* Side Node List - NOT Saturday Night Live! */

      std::set<int> ss_side_nodes;
      std::set<int> ss_side_nodes_global;
      for (int es = 0; es < exo->ss_num_sides[i]; es++) {
        for (int q = exo->ss_elem_index[i]; q < (exo->ss_elem_index[i] + exo->ss_num_sides[i]);
             q++) {
          int elem = exo->ss_elem_list[q];
          int side = exo->ss_side_list[q];
          int num_nodes = build_side_node_list(elem, side - 1, exo, snl);
          for (int j = 0; j < num_nodes; j++) {
            ss_side_nodes.insert(snl[j]);
            if (exo_to_global.find(snl[j]) != exo_to_global.end()) {
              ss_side_nodes_global.insert(mesh->globals(0)[exo_to_global[snl[j]]]);
              if (verbose)
                std::cout << "node found " << mesh->globals(0)[exo_to_global[snl[j]]] << " "
                          << dpi->node_index_global[snl[j]] << "\n";
            } else {
              if (verbose)
                std::cout << "node not found " << dpi->node_index_global[snl[j]] << "\n";
            }
          }
        }
      }

      if (verbose) {
        std::cout << "ss#" << dpi->ss_id_global[i] << " Proc " << ProcID << " ss_side_nodes ";

        for (auto n : ss_side_nodes) {
          std::cout << n << " ";
        }
        std::cout << "\n";
      }

      int nnodes = 0;
      for (int side = 0; side < nsides; side++) {
        nnodes += exo->ss_node_cnt_list[i][side];
      }
      int nnodes_owned = 0;
      for (int side = 0; side < nnodes; side++) {
        //        int exoindex = exo->ss_node_list[i][side];
        //        if (exo_to_global.find(exoindex) != exo_to_global.end()) {
        nnodes_owned++;
        //        }
      }

      int max_nnodes;
      MPI_Allreduce(&nnodes_owned, &max_nnodes, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      std::vector<int> all_owned(max_nnodes * Num_Proc);
      std::vector<int> my_owned(max_nnodes);
      std::fill(my_owned.begin(), my_owned.end(), -1);

      HostWrite<LO> h_set_nodes_q(nnodes_owned);
      int index = 0;
      for (int idx = 0; idx < nnodes; idx++) {
        int exoindex = exo->ss_node_list[i][idx];
        //        if (exo_to_global.find(exoindex) != exo_to_global.end()) {
        h_set_nodes_q[index] = dpi->node_index_global[exoindex];
        index++;
        //        }
      }
      for (int i = 0; i < index; i++) {
        my_owned[i] = h_set_nodes_q[i];
      }
      std::set<int> my_owned_set(my_owned.begin(), my_owned.end());

      MPI_Allgather(my_owned.data(), max_nnodes, MPI_INT, all_owned.data(), max_nnodes, MPI_INT,
                    MPI_COMM_WORLD);
      std::set<int> owned_set(all_owned.begin(), all_owned.end());
      std::vector<int> my_nodes_sorted(mesh->nverts());
      for (int i = 0; i < mesh->nverts(); i++) {
        my_nodes_sorted[i] = mesh->globals(0)[i];
      }
      std::sort(my_nodes_sorted.begin(), my_nodes_sorted.end());
      std::vector<int> all_owned_set(owned_set.begin(), owned_set.end());
      std::vector<int> mark_nodes;
      mark_nodes.reserve(all_owned_set.size());
      for (size_t j = 0; j < all_owned_set.size(); j++) {
        if (all_owned_set[j] != -1 &&
            global_to_index.find(all_owned_set[j]) != global_to_index.end()) {
          // TODO
          mark_nodes.push_back(global_to_index[all_owned_set[j]]);
        }
      }

      HostWrite<LO> h_set_nodes(mark_nodes.size());
      for (size_t j = 0; j < mark_nodes.size(); j++) {
        h_set_nodes[j] = mark_nodes[j];
      }

      auto set_nodes2nodes = LOs(h_set_nodes.write());
      auto nodes_are_in_set = mark_image(set_nodes2nodes, mesh->nverts());
      auto sides_are_in_set = mark_up_all(mesh, VERT, dim - 1, nodes_are_in_set);
      auto set_sides2side_tmp = collect_marked(sides_are_in_set);
      int max_sides;
      int my_size = set_sides2side_tmp.size();
      MPI_Allreduce(&my_size, &max_sides, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      std::vector<int> send_sides(max_sides);
      std::fill(send_sides.begin(), send_sides.end(), -1);
      for (int j = 0; j < my_size; j++) {
        send_sides[j] = mesh->globals(dim - 1)[set_sides2side_tmp[j]];
      }
      std::vector<int> all_sides(max_sides * Num_Proc);
      MPI_Allgather(send_sides.data(), max_sides, MPI_INT, all_sides.data(), max_sides, MPI_INT,
                    MPI_COMM_WORLD);

      std::vector<int> sides;

      for (size_t j = 0; j < all_sides.size(); j++) {
        if (global_to_index_sides.find(all_sides[j]) != global_to_index_sides.end()) {
          sides.push_back(global_to_index_sides[all_sides[j]]);
        }
      }
      std::sort(sides.begin(), sides.end());

      if (verbose)
        std::cout << "Proc" << ProcID << " has found " << sides.size() << "sides\n";
      Write<LO> set_sides2side(sides.size());

      for (size_t j = 0; j < sides.size(); j++) {
        set_sides2side[j] = sides[j];
      }

      auto surface_id = dpi->ss_id_global[i];
      if (verbose) {
        std::cout << "P" << ProcID << " side set #" << surface_id << " \"" << name_ptrs[i]
                  << "\" has " << nsides << " sides, will be surface " << surface_id << "\n";

        std::cout << "P" << ProcID << " side set #" << surface_id << " nodes = ";
        for (int i = 0; i < h_set_nodes.size(); i++) {
          std::cout << mesh->globals(0).data()[h_set_nodes.data()[i]] << " ";
        }
        std::cout << "P" << ProcID << " side set #" << surface_id << " sides = ";
        for (int i = 0; i < set_sides2side.size(); i++) {
          std::cout << mesh->globals(dim - 1).data()[set_sides2side.data()[i]] << " ";
        }
        std::cout << "\n\n\n";
      }
      map_value_into(surface_id, set_sides2side, side_class_ids_w);
      map_value_into(I8(dim - 1), set_sides2side, side_class_dims_w);
      mesh->class_sets[name_ptrs[i]].push_back({I8(dim - 1), surface_id});
    }
    for (int offset = 0; offset < dpi->num_side_sets_global; offset++) {
      auto surface_id = dpi->ss_id_global[offset];
      mesh->class_sets[name_ptrs[offset]].push_back({I8(dim - 1), surface_id});
    }
  }
  auto elem_class_ids = LOs(elem_class_ids_w);
  auto side_class_ids = LOs(side_class_ids_w);
  auto side_class_dims = Read<I8>(side_class_dims_w);
  mesh->add_tag(dim, "class_id", 1, elem_class_ids);
  mesh->add_tag(dim - 1, "class_id", 1, side_class_ids);
  mesh->set_tag(dim - 1, "class_dim", side_class_dims);
  /*
  classify_elements(mesh);
  auto elem_class_ids = LOs(elem_class_ids_w);
  //auto side_class_ids = LOs(side_class_ids_w);
  //auto side_class_dims = Read<I8>(side_class_dims_w);
//  mesh->add_tag(dim, "class_id", 1, elem_class_ids);
  mesh->set_parting(OMEGA_H_GHOSTED);
  auto sides_are_exposed = mark_exposed_sides(mesh);
  classify_sides_by_exposure(mesh, sides_are_exposed);
  //mesh->add_tag(dim - 1, "class_id", 1, side_class_ids);
  //mesh->set_tag(dim - 1, "class_dim", side_class_dims);
  finalize_classification(mesh);
//  mesh->balance();
//  mesh->set_parting(OMEGA_H_GHOSTED);
   */
  classify_elements(mesh);
  //  auto elem_class_ids = LOs(elem_class_ids_w);
  // auto side_class_ids = LOs(side_class_ids_w);
  // auto side_class_dims = Read<I8>(side_class_dims_w);
  //  mesh->add_tag(dim, "class_id", 1, elem_class_ids);
  mesh->set_parting(OMEGA_H_GHOSTED);
  //  auto sides_are_exposed = mark_exposed_sides(mesh);
  //  classify_sides_by_exposure(mesh, sides_are_exposed);
  // mesh->add_tag(dim - 1, "class_id", 1, side_class_ids);
  // mesh->set_tag(dim - 1, "class_dim", side_class_dims);
  finalize_classification(mesh);
}

void convert_omega_h_to_goma(
    const char *path, Mesh *mesh, Exo_DB *exo, Dpi *dpi, bool verbose, int classify_with) {

  // The adapted partition is collected in m, then used to set up exo and dpi
  // in the same form read_mesh_exoII() would produce from a Nemesis file
  mesh->set_parting(OMEGA_H_ELEM_BASED);
  std::set<LO> region_set;
  auto dim = mesh->dim();
  auto title = "Omega_h " OMEGA_H_SEMVER " Exodus Output";
  AdaptedMesh m;
  m.title = title;
  m.dim = dim;

  auto class_sets = mesh->class_sets;
  // TODO multiblock
  std::set<LO> surface_set;
  for (auto &it : class_sets) {
    auto key = it.first;
    auto value = it.second;
    for (size_t i = 0; i < value.size(); i++) {
      if (static_cast<int>(i) < dpi->num_side_sets_global) {
        surface_set.insert(value[i].id);
      }
    }
  }
  std::vector<LO> surface_set_vec(surface_set.begin(), surface_set.end());
  auto elem_class_ids = mesh->get_array<ClassId>(dim, "class_id");
  auto h_elem_class_ids = HostRead<LO>(elem_class_ids);
  for (LO i = 0; i < h_elem_class_ids.size(); ++i) {
    region_set.insert(h_elem_class_ids[i]);
  }
  auto side_class_ids = mesh->get_array<ClassId>(dim - 1, "class_id");
  auto side_class_dims = mesh->get_array<I8>(dim - 1, "class_dim");
  auto h_side_class_ids = HostRead<LO>(side_class_ids);
  auto h_side_class_dims = HostRead<I8>(side_class_dims);

  auto all_conn = mesh->ask_elem_verts();
  auto deg = element_degree(mesh->family(), dim, VERT);
  auto vert_owners = mesh->ask_owners(0).ranks;

  // optimize later if this is slow but make a list of all nodes that we don't own and list
  // them as needed, then use that to populate a communication setup

  std::vector<int> my_global_comm_nodes;
  std::unordered_map<int, int> global_node_to_local;
  for (int i = 0; i < mesh->nverts(); i++) {
    if (vert_owners[i] != ProcID) {
      my_global_comm_nodes.push_back(mesh->globals(0)[i]);
    }
    global_node_to_local.insert(std::make_pair((int)mesh->globals(0)[i], i));
  }
  std::sort(my_global_comm_nodes.begin(), my_global_comm_nodes.end());
  int comm_size = my_global_comm_nodes.size();
  int max_comm_size;
  MPI_Allreduce(&comm_size, &max_comm_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  std::vector<int> all_neighbors_comm_nodes(max_comm_size * Num_Proc);

  my_global_comm_nodes.resize(max_comm_size);
  for (int i = comm_size; i < max_comm_size; i++) {
    my_global_comm_nodes[i] = -1;
  }

  MPI_Allgather(my_global_comm_nodes.data(), max_comm_size, MPI_INT,
                all_neighbors_comm_nodes.data(), max_comm_size, MPI_INT, MPI_COMM_WORLD);

  // update our comm nodes with nodes we find in the neighbors comm nodes
  std::set<int> updated_comm_nodes_set;
  for (auto node : all_neighbors_comm_nodes) {
    if (global_node_to_local.find(node) != global_node_to_local.end()) {
      updated_comm_nodes_set.insert(node);
    }
  }
  std::vector<int> updated_comm_nodes(updated_comm_nodes_set.begin(), updated_comm_nodes_set.end());
  comm_size = updated_comm_nodes.size();
  MPI_Allreduce(&comm_size, &max_comm_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  updated_comm_nodes.resize(max_comm_size);
  for (int i = comm_size; i < max_comm_size; i++) {
    updated_comm_nodes[i] = -1;
  }
  all_neighbors_comm_nodes.resize(max_comm_size * Num_Proc);
  MPI_Allgather(updated_comm_nodes.data(), max_comm_size, MPI_INT, all_neighbors_comm_nodes.data(),
                max_comm_size, MPI_INT, MPI_COMM_WORLD);

  // With element based parting every shared node is a border node on each
  // processor that has it, so a neighbor is any other processor that shares at
  // least one node with us. This must match the node comm map construction below.
  // (Selecting neighbors by node owner misses processors that only share a node
  // whose owner is a third processor, e.g. at partition triple points.)
  std::vector<int> neighbor_list;
  for (int proc = 0; proc < Num_Proc; proc++) {
    if (proc == ProcID)
      continue;
    int offset = proc * max_comm_size;
    for (int i = 0; i < max_comm_size; i++) {
      int gnode = all_neighbors_comm_nodes[offset + i];
      if (gnode != -1 && global_node_to_local.find(gnode) != global_node_to_local.end()) {
        neighbor_list.push_back(proc);
        break;
      }
    }
  }

  std::vector<int> proc_node_counts(neighbor_list.size());
  std::vector<std::vector<int>> proc_node_list(neighbor_list.size());
  std::vector<int> proc_node_idx(neighbor_list.size() + 1);
  std::fill(proc_node_counts.begin(), proc_node_counts.end(), 0);
  std::set<int> boundary_nodes;
  for (int proc = 0; proc < Num_Proc; proc++) {
    int offset = proc * max_comm_size;
    if (proc == ProcID)
      continue;
    for (int i = 0; i < max_comm_size; i++) {
      int gnode = all_neighbors_comm_nodes[offset + i];
      if (gnode != -1) {
        if (global_node_to_local.find(gnode) != global_node_to_local.end()) {
          auto it = std::find(neighbor_list.begin(), neighbor_list.end(), proc);
          if (it == neighbor_list.end()) {
            GOMA_EH(GOMA_ERROR,
                    "Omega_h conversion: processor %d shares nodes but is not a neighbor", proc);
          }
          int index = it - neighbor_list.begin();
          proc_node_list[index].push_back(global_node_to_local[gnode]);
          proc_node_counts[index] += 1;
          boundary_nodes.insert(global_node_to_local[gnode]);
        }
      }
    }
  }

  std::vector<bool> keep_nodes(mesh->nverts());
  std::fill(keep_nodes.begin(), keep_nodes.end(), true);

  std::vector<int> mesh_owned_elements;
  for (int i = 0; i < mesh->nelems(); i++) {
    mesh_owned_elements.push_back(i);
  }

  proc_node_idx[0] = 0;
  std::vector<int> node_cmap_ids;
  std::vector<int> node_cmap_to_neighbor;
  for (unsigned int i = 0; i < neighbor_list.size(); i++) {
    proc_node_idx[i + 1] = proc_node_idx[i] + proc_node_counts[i];
    if (proc_node_counts[i] > 0) {
      node_cmap_ids.push_back(neighbor_list[i]);
      node_cmap_to_neighbor.push_back(i);
    }
  }
  std::vector<int> boundary_nodes_sorted(boundary_nodes.begin(), boundary_nodes.end());
  std::sort(boundary_nodes_sorted.begin(), boundary_nodes_sorted.end());
  std::vector<int> internal_nodes;
  for (int j = 0; j < vert_owners.size(); j++) {
    if (vert_owners[j] == ProcID) {
      if (!std::binary_search(boundary_nodes_sorted.begin(), boundary_nodes_sorted.end(), j)) {
        internal_nodes.push_back(j);
      }
    }
  }

  std::vector<int> reduced_conn;
  std::vector<int> global_elem;
  std::vector<int> old_to_new_node_map(mesh->nverts());
  std::vector<int> old_to_new_elem_map(mesh->nelems());
  std::fill(old_to_new_node_map.begin(), old_to_new_node_map.end(), -1);
  std::fill(old_to_new_elem_map.begin(), old_to_new_elem_map.end(), -1);
  std::set<int> new_nodes;
  for (auto offset : mesh_owned_elements) {
    for (int j = 0; j < deg; j++) {
      reduced_conn.push_back(all_conn[offset * deg + j]);
    }
    global_elem.push_back(mesh->globals(mesh->dim())[offset]);
  }
  std::vector<int> new_nodes_v;
  new_nodes_v.reserve(internal_nodes.size() + boundary_nodes_sorted.size());
  for (auto inn : internal_nodes) {
    new_nodes_v.push_back(inn);
  }
  for (auto bnn : boundary_nodes_sorted) {
    new_nodes_v.push_back(bnn);
  }
  std::vector<int> global_node(new_nodes_v.size());
  for (size_t i = 0; i < new_nodes_v.size(); i++) {
    global_node[i] = mesh->globals(0)[new_nodes_v[i]];
    old_to_new_node_map[new_nodes_v[i]] = i;
  }
  for (size_t i = 0; i < mesh_owned_elements.size(); i++) {
    old_to_new_elem_map[mesh_owned_elements[i]] = i;
  }

  for (size_t i = 0; i < reduced_conn.size(); i++) {
    auto old = reduced_conn[i];
    auto new_node = old_to_new_node_map[old];
    assert(new_node != -1);
    reduced_conn[i] = new_node + 1;
  }

  for (size_t i = 0; i < neighbor_list.size(); i++) {
    for (size_t j = 0; j < proc_node_list[i].size(); j++) {
      proc_node_list[i][j] = old_to_new_node_map[proc_node_list[i][j]] + 1;
    }
  }

  m.num_nodes = global_node.size();
  m.num_elems = global_elem.size();

  Few<Write<Real>, 3> coord_blk;
  for (Int i = 0; i < dim; ++i)
    coord_blk[i] = Write<Real>(new_nodes_v.size());
  auto coords = mesh->coords();
  for (size_t i = 0; i < new_nodes_v.size(); i++) {
    auto local_node = new_nodes_v[i];
    for (Int j = 0; j < dim; ++j) {
      coord_blk[j][i] = coords[local_node * dim + j];
    }
  }
  HostRead<Real> h_coord_blk[3];
  for (Int i = 0; i < dim; ++i)
    h_coord_blk[i] = HostRead<Real>(coord_blk[i]);
  for (Int i = 0; i < dim; ++i)
    m.coords[i].assign(h_coord_blk[i].data(), h_coord_blk[i].data() + h_coord_blk[i].size());
  auto elems2file_idx = Write<LO>(mesh->nelems());
  auto elem_file_offset = LO(0);

  // TODO
  assert(region_set.size() == 1);

  for (auto block_id : region_set) {
    auto type_name = (dim == 3) ? "tetra4" : "tri3";
    auto elems_in_block = each_eq_to(elem_class_ids, block_id);
    auto block_elems2elem = collect_marked(elems_in_block);
    auto nblock_elems = block_elems2elem.size();
    if (verbose) {
      std::cout << "element block " << block_id << " has " << nblock_elems << " of type "
                << type_name << '\n';
    }
    m.block_id = block_id;
    m.elem_type = type_name;
    m.nodes_per_elem = deg;
    m.conn = reduced_conn;
    auto f = OMEGA_H_LAMBDA(LO block_elem) {
      elems2file_idx[block_elems2elem[block_elem]] = elem_file_offset + block_elem;
    };
    parallel_for(nblock_elems, f);
    elem_file_offset += nblock_elems;
  }

  std::vector<int> global_node_counts(dpi->num_node_sets_global);
  std::vector<int> sset_global_side;
  if (1 && classify_with) {

    std::vector<std::vector<int>> ns_nodes(surface_set_vec.size());
    for (auto set_id : surface_set_vec) {
      auto sides_in_set =
          land_each(each_eq_to(side_class_ids, set_id), each_eq_to(side_class_dims, I8(dim - 1)));
      if (classify_with & exodus::SIDE_SETS) {
        std::vector<int> sides2elem_new;
        std::vector<int> sides2local_new;
        auto set_sides2side = collect_marked(sides_in_set);
        auto nset_sides = set_sides2side.size();
        if (verbose) {
          std::cout << "side set " << set_id << " has " << nset_sides << " sides\n";
        }
        auto sides2elems = mesh->ask_up(dim - 1, dim);
        Write<int> set_sides2elem(nset_sides);
        Write<int> set_sides2local(nset_sides);
        for (int set_side = 0; set_side < nset_sides; set_side++) {
          auto side = set_sides2side[set_side];
          auto side_elem = sides2elems.a2ab[side];
          auto elem = sides2elems.ab2b[side_elem];
          auto elem_in_file = elems2file_idx[elem];
          auto code = sides2elems.codes[side_elem];
          auto which_down = code_which_down(code);
          set_sides2elem[set_side] = elem_in_file + 1;
          set_sides2local[set_side] = side_osh2exo(dim, which_down);
          if (std::find(mesh_owned_elements.begin(), mesh_owned_elements.end(), elem_in_file) !=
              mesh_owned_elements.end()) {
            sides2elem_new.push_back(old_to_new_elem_map[elem_in_file] + 1);
            sides2local_new.push_back(side_osh2exo(dim, which_down));
          }
        }
        sset_global_side.push_back(sides2elem_new.size());
        m.ss_ids.push_back(set_id);
        m.ss_elems.push_back(sides2elem_new);
        m.ss_sides.push_back(sides2local_new);
      }
      if (classify_with & exodus::NODE_SETS) {
        auto nodes_in_set = mark_down(mesh, dim - 1, VERT, sides_in_set);
        auto set_nodes2node = collect_marked(nodes_in_set);
        auto set_nodes2node_ex = add_to_each(set_nodes2node, 1);
        auto nset_nodes = set_nodes2node.size();
        std::size_t set_id_offset = 0;
        for (std::size_t i = 0; i < surface_set_vec.size(); i++) {
          if (surface_set_vec[i] == set_id) {
            set_id_offset = i;
          }
        }
        for (int i = 0; i < set_nodes2node.size(); i++) {
          int local_node = old_to_new_node_map[set_nodes2node[i]];
          if (local_node != -1) {
            ns_nodes[set_id_offset].push_back(local_node + 1);
          }
        }
        if (verbose) {
          std::cout << "node set " << set_id << " has " << nset_nodes << " nodes\n";
        }
        m.ns_ids.push_back(set_id);
        m.ns_nodes.push_back(ns_nodes[set_id_offset]);
      }
    }

    // Nodeset dpi information needs total nodes for each nodeset and we need
    // to share information in order to have an accurate count.
    std::vector<int> my_sizes(surface_set_vec.size());
    std::vector<int> all_sizes(surface_set_vec.size() * Num_Proc);
    for (size_t i = 0; i < surface_set_vec.size(); i++) {
      my_sizes[i] = ns_nodes[i].size();
    }

    MPI_Allgather(my_sizes.data(), surface_set_vec.size(), MPI_INT, all_sizes.data(),
                  surface_set_vec.size(), MPI_INT, MPI_COMM_WORLD);

    for (size_t i = 0; i < surface_set_vec.size(); i++) {
      int max = 0;
      for (int j = 0; j < Num_Proc; j++) {
        max = std::max(all_sizes[j * surface_set_vec.size() + i], max);
      }
      std::vector<int> ns_global_nodes(max);
      for (size_t j = 0; j < ns_nodes[i].size(); j++) {
        ns_global_nodes[j] = global_node[ns_nodes[i][j] - 1];
      }

      std::vector<int> all_ns_nodes(max * Num_Proc);
      MPI_Allgather(ns_global_nodes.data(), max, MPI_INT, all_ns_nodes.data(), max, MPI_INT,
                    MPI_COMM_WORLD);

      std::set<int> ns_nodes_set;
      for (int j = 0; j < Num_Proc; j++) {
        size_t offset = j * max;
        for (int k = 0; k < all_sizes[j * surface_set_vec.size() + i]; k++) {
          ns_nodes_set.insert(all_ns_nodes[offset + k]);
        }
      }

      global_node_counts[i] = ns_nodes_set.size();
    }
  }
  std::vector<int> node_map(global_node.begin(), global_node.end());
  auto add1 = [](int &v) { v++; };
  std::for_each(node_map.begin(), node_map.end(), add1);
  std::vector<int> elem_map(global_elem.begin(), global_elem.end());
  std::for_each(elem_map.begin(), elem_map.end(), add1);

  m.node_map = node_map;
  m.elem_map = elem_map;

  int max_sets = surface_set_vec.size();
  int gmax_sets;
  MPI_Allreduce(&max_sets, &gmax_sets, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  std::vector<int> all_sets(gmax_sets * Num_Proc + 1);
  std::fill(all_sets.begin(), all_sets.end(), -1);
  std::vector<int> my_sets(surface_set_vec.begin(), surface_set_vec.end());
  my_sets.resize(gmax_sets);
  MPI_Allgather(my_sets.data(), my_sets.size(), MPI_INT, all_sets.data(), gmax_sets, MPI_INT,
                MPI_COMM_WORLD);
  std::set<int> global_set_uniq;
  for (auto c : all_sets) {
    if (c != -1) {
      global_set_uniq.insert(c);
    }
  }

  std::vector<int> global_sets(global_set_uniq.begin(), global_set_uniq.end());
  std::sort(global_sets.begin(), global_sets.end());
  std::vector<int> global_side_counts(global_sets.size());

  for (size_t setidx = 0; setidx < global_sets.size(); setidx++) {
    auto set = global_sets[setidx];
    auto my_offset = std::find(my_sets.begin(), my_sets.end(), set);
    if (my_offset != my_sets.end()) {
      int index = std::distance(my_sets.begin(), my_offset);
      MPI_Allreduce(&sset_global_side[index], &global_side_counts[setidx], 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
    }
  }
  std::vector<int> eb_id(1);
  eb_id[0] = 1;
  std::vector<int> eb_counts(1);
  eb_counts[0] = mesh->nglobal_ents(mesh->dim());

  m.num_nodes_global = mesh->nglobal_ents(0);
  m.num_elems_global = mesh->nglobal_ents(mesh->dim());
  m.global_set_ids = global_sets;
  m.global_ns_node_counts.assign(global_sets.size(), 0);
  for (size_t i = 0; i < global_sets.size() && i < global_node_counts.size(); i++) {
    m.global_ns_node_counts[i] = global_node_counts[i];
  }
  m.global_ss_side_counts = global_side_counts;
  m.global_eb_ids = eb_id;
  m.global_eb_counts = eb_counts;
  m.num_internal_nodes = internal_nodes.size();
  m.num_boundary_nodes = boundary_nodes_sorted.size();
  m.num_internal_elems = global_elem.size();
  m.node_cmap_ids = node_cmap_ids;
  for (auto nidx : node_cmap_to_neighbor) {
    m.node_cmap_nodes.push_back(proc_node_list[nidx]);
    m.node_cmap_procs.emplace_back(proc_node_list[nidx].size(), neighbor_list[nidx]);
  }

  for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
    free(idv[imtrx]);
  }
  free(idv);
  idv = NULL;

  // Sizes of the old mesh, needed to free per node arrays below; exo and dpi
  // are reset before those arrays are freed.
  const int old_num_nodes = exo->num_nodes;
  const int old_num_total_nodes =
      dpi->num_internal_nodes + dpi->num_boundary_nodes + dpi->num_external_nodes;

  free_Surf_BC(First_Elem_Side_BC_Array, exo);
  free_Edge_BC(First_Elem_Edge_BC_Array, exo, dpi);
  free_nodes();
  if (Num_Proc == 1) {
    free_dpi_uni(dpi);
  } else {
    free_dpi(dpi);
  }
  free_exo(exo);
  init_exo_struct(exo);
  init_dpi_struct(dpi);

  if (rotation_allocated) {
    // rotation is allocated per node in rotate.c with num_total_nodes entries,
    // local_ROT_list is only allocated by one of the two allocation paths
    for (int i = 0; i < old_num_total_nodes; i++) {
      for (int j = 0; j < NUM_VECTOR_EQUATIONS; j++) {
        if (rotation[i][j] != NULL) {
          for (int k = 0; k < pd_glob[0]->Num_Dim; k++) {
            free(rotation[i][j][k]);
          }
          free(rotation[i][j]);
        }
      }
      free(rotation[i]);
      if (local_ROT_list != NULL) {
        free(local_ROT_list[i]);
      }
    }
    free(rotation);
    rotation = NULL;
    free(local_ROT_list);
    local_ROT_list = NULL;
    rotation_allocated = FALSE;
  }

  if (goma_automatic_rotations.rotation_nodes != NULL) {
    for (int i = 0; i < old_num_nodes; i++) {
      // for (int j = 0; j < GOMA_MAX_NORMALS_PER_NODE; j++) {
      //   gds_vector_free((goma_automatic_rotations.rotation_nodes)[i].normals[j]);
      //   gds_vector_free((goma_automatic_rotations.rotation_nodes)[i].average_normals[j]);
      //   gds_vector_free((goma_automatic_rotations.rotation_nodes)[i].tangent1s[j]);
      //   gds_vector_free((goma_automatic_rotations.rotation_nodes)[i].tangent2s[j]);
      // }
      for (int j = 0; j < DIM; j++) {
        goma_normal_free((goma_automatic_rotations.rotation_nodes)[i].rotated_coord[j]);
      }
    }
    free(goma_automatic_rotations.rotation_nodes);
    goma_automatic_rotations.rotation_nodes = NULL;
  }

  //  const char * tmpfile = "";
  // The adapted mesh is not read from a file, ExoFile now refers to the
  // adapted mesh written below
  strncpy(ExoFile, path, MAX_FNL - 1);
  strncpy(ExoFileOutMono, path, 127);
  strncpy(ExoFileOut, path, 127);
  multiname(ExoFileOut, ProcID, Num_Proc);

  for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
    for (int i = 0; i < old_num_total_nodes; i++) {
      free(Local_Offset[imtrx][i]);
      free(Dolphin[imtrx][i]);
    }
    free(Dolphin[imtrx]);
    free(Local_Offset[imtrx]);
  }
  safer_free((void **)&Local_Offset);
  safer_free((void **)&Dolphin);

  setup_mesh_in_memory(exo, dpi, m, path);
  one_base(exo, Num_Proc);
  wr_mesh_exo(exo, ExoFileOut, 0);
  zero_base(exo);

  if (Num_Proc > 1) {
    wr_dpi(dpi, ExoFileOut);
  }
  dpi->exodus_to_omega_h_node = (int *)malloc(sizeof(int) * global_node.size());
  for (size_t i = 0; i < old_to_new_node_map.size(); i++) {
    if (old_to_new_node_map[i] != -1) {
      dpi->exodus_to_omega_h_node[old_to_new_node_map[i]] = i;
    }
  }
  if (dpi->num_neighbors > 0) {
    for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
      free(cx[imtrx]);
      cx[imtrx] = alloc_struct_1(Comm_Ex, DPI_ptr->num_neighbors);
    }
    // Request and Status are shared by all matrices
    free(Request);
    free(Status);
    Request = alloc_struct_1(MPI_Request, Num_Requests * DPI_ptr->num_neighbors);
    Status = alloc_struct_1(MPI_Status, Num_Requests * DPI_ptr->num_neighbors);
  }

  end_code();
}
} // namespace exodus
} // namespace goma

void adapt_mesh(Omega_h::Mesh &mesh) {
  Omega_h::MetricInput genopts;
  //  genopts.sources.push_back(
  //      Omega_h::MetricSource{OMEGA_H_VARIATION, 1e-3, "phi", OMEGA_H_ISO_SIZE});
  genopts.sources.push_back(Omega_h::MetricSource{OMEGA_H_GIVEN, 1.0, "iso_size_metric",
                                                  OMEGA_H_ISO_SIZE, OMEGA_H_ABSOLUTE});
  //      genopts.sources.push_back(Omega_h::MetricSource{OMEGA_H_GIVEN, 1.0,
  //    "initial_metric",OMEGA_H_ISO_SIZE,OMEGA_H_ABSOLUTE});
  genopts.should_limit_lengths = true;
  genopts.min_length = 1e-6;
  genopts.max_length = 0.6;
  genopts.should_limit_gradation = true;
  genopts.max_gradation_rate = 0.3;
  Omega_h::add_implied_isos_tag(&mesh);
  Omega_h::generate_target_metric_tag(&mesh, genopts);

  Omega_h::AdaptOpts opts(&mesh);
  for (int j = V_FIRST; j < V_LAST; j++) {
    int imtrx = upd->matrix_index[j];
    if (imtrx >= 0) {
      if (j == MASS_FRACTION) {
        for (int mf = 0; mf < upd->Max_Num_Species; mf++) {
          std::string species_name = Exo_Var_Names[j].name2 + std::to_string(mf);
          opts.xfer_opts.type_map[species_name] = OMEGA_H_LINEAR_INTERP;
        }
      } else {
        opts.xfer_opts.type_map[Exo_Var_Names[j].name2] = OMEGA_H_LINEAR_INTERP;
      }
    }
  }
  for (int w = 0; w < efv->Num_external_field; w++) {
    opts.xfer_opts.type_map[efv->name[w]] = OMEGA_H_LINEAR_INTERP;
  }
  opts.max_length_allowed = 5;
  opts.max_length_desired = 1.8;
  opts.should_coarsen_slivers = true;
  opts.should_refine = true;
  opts.min_quality_desired = 0.5;
  int count = 1000;

  for (int i = 0; i < count; i++) {
    if (Omega_h::approach_metric(&mesh, opts)) {
      if (ProcID == 0) {
        std::cout << "Omega_h approach_metric adapt count = " << i << "\n";
      }
      Omega_h::adapt(&mesh, opts);
    } else {
      break;
    }
  }
  auto imb = mesh.imbalance();
  if (ProcID == 0) {
    std::cout << "Mesh imbalance = " << imb << "\n";
  }
  if (power_of_two(Num_Proc)) {
    mesh.balance();
    imb = mesh.imbalance();
    if (ProcID == 0) {
      std::cout << "Mesh imbalance after balance = " << imb << "\n";
    }
  } else {
    GOMA_WH_MANY(GOMA_ERROR,
                 "Omega_h requires a power of 2 number of processors to balance the mesh");
    if (imb > 2.5) {
      GOMA_EH(-1, "Mesh imbalance too large: %g exiting", imb);
    }
  }
}

extern "C" {

void copy_solution(Exo_DB *exo, Dpi *dpi, double **x, Omega_h::Mesh &mesh) {
  for (int j = V_FIRST; j < V_LAST; j++) {
    int imtrx = upd->matrix_index[j];
    if (imtrx >= 0) {
      if (j == MASS_FRACTION) {
        for (int mf = 0; mf < upd->Max_Num_Species; mf++) {
          std::string species_name = Exo_Var_Names[j].name2 + std::to_string(mf);
          auto var_values = mesh.get_array<Omega_h::Real>(Omega_h::VERT, species_name);

          for (int i = 0; i < (dpi->num_internal_nodes + dpi->num_boundary_nodes); i++) {
            int ja = Index_Solution(i, j, mf, 0, -2, imtrx);
            GOMA_EH(ja, "could not find solution");
            int index = i;
            if (dpi->exodus_to_omega_h_node != NULL) {
              index = dpi->exodus_to_omega_h_node[exo->ghost_node_to_base[i]];
            }
            x[imtrx][ja] = var_values[index];
          }
        }
      } else {
        auto var_values = mesh.get_array<Omega_h::Real>(Omega_h::VERT, Exo_Var_Names[j].name2);

        for (int i = 0; i < (dpi->num_internal_nodes + dpi->num_boundary_nodes); i++) {
          int ja = Index_Solution(i, j, 0, 0, -2, imtrx);
          GOMA_EH(ja, "could not find solution");
          int index = i;
          if (dpi->exodus_to_omega_h_node != NULL) {
            index = dpi->exodus_to_omega_h_node[exo->ghost_node_to_base[i]];
          }
          x[imtrx][ja] = var_values[index];
        }
      }
    }
  }
  for (int w = 0; w < efv->Num_external_field; w++) {
    auto var_values = mesh.get_array<Omega_h::Real>(Omega_h::VERT, efv->name[w]);
    for (int i = 0; i < (dpi->num_internal_nodes + dpi->num_boundary_nodes); i++) {
      int index = i;
      if (dpi->exodus_to_omega_h_node != NULL) {
        index = dpi->exodus_to_omega_h_node[exo->ghost_node_to_base[i]];
      }
      efv->ext_fld_ndl_val[w][i] = var_values[index];
    }
  }
}

// start with just level set field
void adapt_mesh_omega_h(struct GomaLinearSolverData **ams,
                        Exo_DB *exo,
                        Dpi *dpi,
                        double **x,
                        double **x_old,
                        double **x_older,
                        double **xdot,
                        double **xdot_old,
                        double **x_oldest,
                        double **resid_vector,
                        double **x_update,
                        double **scale,
                        int step) {

  static std::string base_name;
  static bool first_call = true;
#ifndef OMEGA_H_USE_MPI
  if (Num_Proc > 1) {
    GOMA_EH(GOMA_ERROR, "Parallel mesh adaptation requires Omega_h built with "
                        "-DOmega_h_USE_MPI=ON; rebuild Omega_h or run in serial");
  }
#endif
  int argc = 0;
  char argv[1][8];
  char **argvptr = (char **)argv;
  auto lib = Omega_h::Library(&argc, &argvptr);
  auto classify_with = goma::exodus::NODE_SETS | goma::exodus::SIDE_SETS;
  auto verbose = false;
#ifdef DEBUG_OMEGA_H
  verbose = true;
#endif
  Omega_h::Mesh mesh(&lib);
  goma::exodus::convert_goma_to_omega_h(exo, dpi, x, &mesh, verbose);
  adapt_mesh(mesh);

  if (first_call) {
    base_name = std::string(ExoFileOutMono);
    first_call = false;
  }

  std::stringstream ss2;

  if (step == 0) {
    ss2 << base_name;
  } else if (step > 0) {
    ss2 << base_name << "-s." << step - 1;
  } else {
    GOMA_EH(GOMA_ERROR, "Adapt step given < 0: %d", step);
  }

  goma::exodus::convert_omega_h_to_goma(ss2.str().c_str(), &mesh, exo, dpi, verbose, classify_with);

  resetup_problem(exo, dpi);

  for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
    int numProcUnknowns = NumUnknowns[imtrx] + NumExtUnknowns[imtrx];
    realloc_dbl_1(&x[imtrx], numProcUnknowns, 0);
    realloc_dbl_1(&x_old[imtrx], numProcUnknowns, 0);
    realloc_dbl_1(&x_older[imtrx], numProcUnknowns, 0);
    realloc_dbl_1(&x_update[imtrx], numProcUnknowns, 0);
    realloc_dbl_1(&xdot[imtrx], numProcUnknowns, 0);
    realloc_dbl_1(&xdot_old[imtrx], numProcUnknowns, 0);
    realloc_dbl_1(&x_oldest[imtrx], numProcUnknowns, 0);
    realloc_dbl_1(&resid_vector[imtrx], numProcUnknowns, 0);
    realloc_dbl_1(&scale[imtrx], numProcUnknowns, 0);
    realloc_dbl_1(&x_update[imtrx], numProcUnknowns + numProcUnknowns, 0);
    pg->matrices[imtrx].ams = ams[imtrx];
    pg->matrices[imtrx].x = x[imtrx];
    pg->matrices[imtrx].x_old = x_old[imtrx];
    pg->matrices[imtrx].x_older = x_older[imtrx];
    pg->matrices[imtrx].xdot = xdot[imtrx];
    pg->matrices[imtrx].xdot_old = xdot_old[imtrx];
    pg->matrices[imtrx].x_update = x_update[imtrx];
    pg->matrices[imtrx].scale = scale[imtrx];
    pg->matrices[imtrx].resid_vector = resid_vector[imtrx];
  }

  for (int w = 0; w < efv->Num_external_field; w++) {
    realloc_dbl_1(&efv->ext_fld_ndl_val[w],
                  dpi->num_internal_nodes + dpi->num_boundary_nodes + dpi->num_external_nodes, 0);
  }
  resetup_matrix(ams, exo, dpi);
  copy_solution(exo, dpi, x, mesh);
  step++;
}

} // extern "C"
#endif

// vim: expandtab sw=2 ts=8
