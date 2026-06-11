#include "adapt/adapt_mmg.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <mmg/mmg2d/libmmg2d.h>
#include <mmg/mmg3d/libmmg3d.h>
#include <mpi.h>
#include <nanoflann.hpp>
#include <optional>
#include <sstream>
#include <stdio.h>
#include <unordered_set>
#include <vector>
extern "C" {
#include "base_mesh.h"
#include "brkfix/fix.h"
#include "dp_types.h"
#include "dpi.h"
#include "el_elm.h"
#include "exo_conn.h"
#include "exo_struct.h"
#include "metis_decomp.h"
#include "mm_as.h"
#include "mm_as_structs.h"
#include "mm_bc.h"
#include "mm_eh.h"
#include "rd_mesh.h"
#include "rf_allo.h"
#include "rf_bc.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_io.h"
#include "rf_io_const.h"
#include "rf_io_structs.h"
#include "rf_mp.h"
#include "rf_util.h"
#include "std.h"
#include <exodusII.h>
#include <limits.h>
#define IGNORE_CPP_DEFINE
#include "sl_util_structs.h"
#undef IGNORE_CPP_DEFINE
#include "adapt/resetup_problem.h"
#include "el_elm.h"
#include "el_elm_info.h"
#include "rd_dpi.h"
#include "rd_exo.h"
#include "rd_mesh.h"
#include "rf_node_const.h"
#include "util/goma_normal.h"
#include "wr_dpi.h"
#include "wr_exo.h"

#define CHECK_EX_ERROR(err, format, ...)                              \
  do {                                                                \
    if (err < 0) {                                                    \
      goma_eh(GOMA_ERROR, __FILE__, __LINE__, format, ##__VA_ARGS__); \
    }                                                                 \
  } while (0)

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

// bool triangle_linear_interp(double xp,
// double yp,
// double xa,
// double ya,
// double xb,
// double yb,
// double xc,
// double yc,
// double &w1,
// double &w2,
// double &w3) {
// const double denom = xa*(yb - yc) + xb*(yb - ya) + xc*(ya - yb);
// const double scale =
// std::max({std::abs(xa - xc), std::abs(xb - xc), std::abs(ya - yc), std::abs(yb - yc), 1.0});
// const double tol = 1e-12 * scale * scale;
// if (std::abs(denom) <= tol) {
// return {};
// }
// w1 = (xp*(yb - yc) + xb*(yc - yp) + xc*(yp - yb)) / denom;
// w2 = (xa*(yp - yc) + xp*(yc - ya) + xc*(ya - yp)) / denom;
// w3 = (xa*(yb - yp) + xb*(yp - ya) + xp*(ya - yb)) / denom;
// if (w1 < -tol || w2 < -tol || w3 < -tol) {
// return false;
// }
// return true;
// }
std::optional<std::array<double, 3>> triangle_linear_interp(
    double x, double y, double x1, double y1, double x2, double y2, double x3, double y3) {
  const double denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
  const double scale =
      std::max({std::abs(x1 - x3), std::abs(x2 - x3), std::abs(y1 - y3), std::abs(y2 - y3), 1.0});
  const double tol = 1e-12 * scale * scale;
  if (std::abs(denom) <= tol) {
    return {};
  }
  double w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom;
  double w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom;
  double w3 = 1.0 - w1 - w2;
  if (w1 < -tol || w2 < -tol || w3 < -tol) {
    return {};
  }
  return std::array<double, 3>{w1, w2, w3};
}

bool tetrahedron_linear_interp(double x,
                               double y,
                               double z,
                               double x1,
                               double y1,
                               double z1,
                               double x2,
                               double y2,
                               double z2,
                               double x3,
                               double y3,
                               double z3,
                               double x4,
                               double y4,
                               double z4,
                               double &w1,
                               double &w2,
                               double &w3,
                               double &w4) {
  auto det3 = [](double a11, double a12, double a13, double a21, double a22, double a23, double a31,
                 double a32, double a33) {
    return a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) +
           a13 * (a21 * a32 - a22 * a31);
  };

  const double det_tet =
      det3(x1 - x4, x2 - x4, x3 - x4, y1 - y4, y2 - y4, y3 - y4, z1 - z4, z2 - z4, z3 - z4);
  const double scale = std::max({std::abs(x1 - x4), std::abs(x2 - x4), std::abs(x3 - x4),
                                 std::abs(y1 - y4), std::abs(y2 - y4), std::abs(y3 - y4),
                                 std::abs(z1 - z4), std::abs(z2 - z4), std::abs(z3 - z4), 1.0});
  const double tol = 1e-12 * scale * scale * scale;

  if (std::abs(det_tet) <= tol) {
    return false;
  }

  w1 = det3(x - x4, x2 - x4, x3 - x4, y - y4, y2 - y4, y3 - y4, z - z4, z2 - z4, z3 - z4) / det_tet;
  w2 = det3(x1 - x4, x - x4, x3 - x4, y1 - y4, y - y4, y3 - y4, z1 - z4, z - z4, z3 - z4) / det_tet;
  w3 = det3(x1 - x4, x2 - x4, x - x4, y1 - y4, y2 - y4, y - y4, z1 - z4, z2 - z4, z - z4) / det_tet;
  w4 = 1.0 - w1 - w2 - w3;

  if (w1 < -tol || w2 < -tol || w3 < -tol || w4 < -tol) {
    return false;
  }

  return true;
}

template <int pdim> struct PointCloud {
  std::vector<std::array<double, pdim>> pts;

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return pts.size(); }

  // Returns the dim'th component of the idx'th point in the class:
  // Since this is inlined and the "dim" argument is typically an immediate
  // value, the
  //  "if/else's" are actually solved at compile time.
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim == 0)
      return pts[idx][0];
    else if (dim == 1)
      return pts[idx][1];
    else {
      if (pdim == 3)
        return pts[idx][2];
      else
        return 0.0;
    }
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned
  //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
  //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX> bool kdtree_get_bbox(BBOX & /* bb */) const { return false; }
};

void interp_solution_to_new_mesh_2d(Exo_DB *exo,
                                    Dpi *dpi,
                                    struct Results_Description **rd,
                                    double **x,
                                    double **xdot,
                                    double time1,
                                    double theta,
                                    double delta_t,
                                    double *new_nodes,
                                    int num_nodes) {
  std::vector<std::array<double, 2>> nodes;
  std::vector<std::array<double, 2>> elements_centroids;
  std::vector<std::pair<int, int>> elem_to_block;
  for (int i = 0; i < exo->num_elem_blocks; i++) {
    for (int j = 0; j < exo->eb_num_elems[i]; j++) {
      elem_to_block.push_back(std::make_pair(i, j));
      int node1_id = exo->eb_conn[i][j * 3];
      int node2_id = exo->eb_conn[i][j * 3 + 1];
      int node3_id = exo->eb_conn[i][j * 3 + 2];
      double x1 = exo->x_coord[node1_id];
      double y1 = exo->y_coord[node1_id];
      double x2 = exo->x_coord[node2_id];
      double y2 = exo->y_coord[node2_id];
      double x3 = exo->x_coord[node3_id];
      double y3 = exo->y_coord[node3_id];
      elements_centroids.push_back({(x1 + x2 + x3) / 3.0, (y1 + y2 + y3) / 3.0});
    }
  }
  PointCloud<2> pc;
  pc.pts = elements_centroids;

  PointCloud<2> pc_points;
  for (int i = 0; i < exo->base_mesh->num_nodes; i++) {
    pc_points.pts.push_back({exo->x_coord[i], exo->y_coord[i]});
  }

  using my_kd_tree_t =
      nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud<2>>,
                                          PointCloud<2>, 2 /* dim */
                                          >;

  my_kd_tree_t index(2, pc, nanoflann::KDTreeSingleIndexAdaptorParams(10));

  my_kd_tree_t index_points(2, pc_points, nanoflann::KDTreeSingleIndexAdaptorParams(10));

  int time_step;
  float ret_float;
  float version;
  char ret_char[3];
  int old_exoid =
      ex_open(ExoFileOutMono, EX_READ, &exo->comp_wordsize, &exo->io_wordsize, &version);
  ex_inquire(old_exoid, EX_INQ_TIME, &time_step, &ret_float, ret_char);

  int num_nodal_vars;
  int err = ex_get_variable_param(old_exoid, EX_NODAL, &num_nodal_vars);
  CHECK_EX_ERROR(err, "ex_get_variable_param");

  std::vector<std::vector<double>> old_values_list(num_nodal_vars);
  for (int var = 0; var < num_nodal_vars; var++) {
    old_values_list[var].resize(exo->num_nodes);
    err = ex_get_var(old_exoid, time_step, EX_NODAL, var + 1, 1, old_values_list[var].size(),
                     old_values_list[var].data());
  }
  ex_close(old_exoid);

  std::vector<std::vector<double>> interpolated_values_list(num_nodal_vars);
  for (int var = 0; var < num_nodal_vars; var++) {
    interpolated_values_list[var].resize(num_nodes);
  }

  for (int i = 0; i < num_nodes; i++) {
    double query_pt[2] = {new_nodes[i * 2], new_nodes[i * 2 + 1]};

    const int num_results = 8;
    std::vector<size_t> ret_index(num_results);
    std::vector<double> out_dist_sqr(num_results);
    nanoflann::KNNResultSet<double> resultSet(num_results);
    resultSet.init(&ret_index[0], &out_dist_sqr[0]);

    index.findNeighbors(resultSet, query_pt, nanoflann::SearchParameters());
    bool found = false;
    // triangular linear interpolation
    for (size_t j = 0; j < resultSet.size(); j++) {
      int block_id = elem_to_block[ret_index[j]].first;
      int elem_id = elem_to_block[ret_index[j]].second;
      int node1_id = exo->eb_conn[block_id][elem_id * 3];
      int node2_id = exo->eb_conn[block_id][elem_id * 3 + 1];
      int node3_id = exo->eb_conn[block_id][elem_id * 3 + 2];
      double x1 = exo->x_coord[node1_id];
      double y1 = exo->y_coord[node1_id];
      double x2 = exo->x_coord[node2_id];
      double y2 = exo->y_coord[node2_id];
      double x3 = exo->x_coord[node3_id];
      double y3 = exo->y_coord[node3_id];
      auto weights = triangle_linear_interp(query_pt[0], query_pt[1], x1, y1, x2, y2, x3, y3);

      if (weights) {
        for (int var = 0; var < num_nodal_vars; var++) {
          found = true;
          double v1 = old_values_list[var][node1_id];
          double v2 = old_values_list[var][node2_id];
          double v3 = old_values_list[var][node3_id];
          interpolated_values_list[var][i] =
              (*weights)[0] * v1 + (*weights)[1] * v2 + (*weights)[2] * v3;
        }
        break;
      }
    }
    if (!found) {
      const int num_results = 1;
      std::vector<size_t> ret_index(num_results);
      std::vector<double> out_dist_sqr(num_results);
      nanoflann::KNNResultSet<double> resultSet(num_results);
      resultSet.init(&ret_index[0], &out_dist_sqr[0]);
      index_points.findNeighbors(resultSet, query_pt, nanoflann::SearchParameters());
      int nearest_node_id = ret_index[0];
      for (int var = 0; var < num_nodal_vars; var++) {
        interpolated_values_list[var][i] = old_values_list[var][nearest_node_id];
      }
    }
  }

  for (int var = 0; var < num_nodal_vars; var++) {
    int status =
        ex_put_var(exo->exoid, 1, EX_NODAL, var + 1, 1, interpolated_values_list[var].size(),
                   interpolated_values_list[var].data());
    GOMA_EH(status, "ex_put_var");
  }

  for (int k = 0; k < efv->Num_external_field; k++) {
    std::vector<double> interpolated_values(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
      double query_pt[2] = {new_nodes[i * 2], new_nodes[i * 2 + 1]};

      const int num_results = 5;
      std::vector<size_t> ret_index(num_results);
      std::vector<double> out_dist_sqr(num_results);
      nanoflann::KNNResultSet<double> resultSet(num_results);
      resultSet.init(&ret_index[0], &out_dist_sqr[0]);

      index.findNeighbors(resultSet, query_pt, nanoflann::SearchParameters());
      bool found = false;
      // triangular linear interpolation
      for (size_t j = 0; j < resultSet.size(); j++) {
        int block_id = elem_to_block[ret_index[j]].first;
        int elem_id = elem_to_block[ret_index[j]].second;
        int node1_id = exo->eb_conn[block_id][elem_id * 3];
        int node2_id = exo->eb_conn[block_id][elem_id * 3 + 1];
        int node3_id = exo->eb_conn[block_id][elem_id * 3 + 2];
        double x1 = exo->x_coord[node1_id];
        double y1 = exo->y_coord[node1_id];
        double x2 = exo->x_coord[node2_id];
        double y2 = exo->y_coord[node2_id];
        double x3 = exo->x_coord[node3_id];
        double y3 = exo->y_coord[node3_id];
        auto weights = triangle_linear_interp(query_pt[0], query_pt[1], x1, y1, x2, y2, x3, y3);

        if (weights) {
          found = true;
          double v1 = efv->ext_fld_ndl_val[k][node1_id];
          double v2 = efv->ext_fld_ndl_val[k][node2_id];
          double v3 = efv->ext_fld_ndl_val[k][node3_id];
          interpolated_values[i] = (*weights)[0] * v1 + (*weights)[1] * v2 + (*weights)[2] * v3;
          break;
        }
      }
      if (!found) {
        const int num_results = 1;
        std::vector<size_t> ret_index(num_results);
        std::vector<double> out_dist_sqr(num_results);
        nanoflann::KNNResultSet<double> resultSet(num_results);
        resultSet.init(&ret_index[0], &out_dist_sqr[0]);
        index_points.findNeighbors(resultSet, query_pt, nanoflann::SearchParameters());
        int nearest_node_id = ret_index[0];
        interpolated_values[i] = efv->ext_fld_ndl_val[k][nearest_node_id];
      }
    }
    realloc_dbl_1(&efv->ext_fld_ndl_val[k], num_nodes, 0);
    std::copy(interpolated_values.begin(), interpolated_values.end(), efv->ext_fld_ndl_val[k]);
  }
}

#if 0
void interp_solution_to_new_mesh_2d(Exo_DB *exo,
                                    Dpi *dpi,
                                    struct Results_Description **rd,
                                    double **x,
                                    double **xdot,
                                    double time1,
                                    double theta,
                                    double delta_t,
                                    double *new_nodes,
                                    int num_nodes) {
  std::vector<std::array<double, 2>> elements_centroids;
  std::vector<std::pair<int, int>> elem_to_block;
  for (int i = 0; i < exo->num_elem_blocks; i++) {
    for (int j = 0; j < exo->eb_num_elems[i]; j++) {
      elem_to_block.push_back(std::make_pair(i, j));
      int node1_id = exo->eb_conn[i][j * 3];
      int node2_id = exo->eb_conn[i][j * 3 + 1];
      int node3_id = exo->eb_conn[i][j * 3 + 2];
      double x1 = exo->x_coord[node1_id];
      double y1 = exo->y_coord[node1_id];
      double x2 = exo->x_coord[node2_id];
      double y2 = exo->y_coord[node2_id];
      double x3 = exo->x_coord[node3_id];
      double y3 = exo->y_coord[node3_id];
      elements_centroids.push_back({(x1 + x2 + x3) / 3.0, (y1 + y2 + y3) / 3.0});
    }
  }

  PointCloud<2> pc;
  pc.pts = elements_centroids;

  PointCloud<2> pc_points;
  for (int i = 0; i < exo->base_mesh->num_nodes; i++) {
    pc_points.pts.push_back({exo->x_coord[i], exo->y_coord[i]});
  }

  using my_kd_tree_2d_t =
      nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud<2>>,
                                          PointCloud<2>, 2 /* dim */>;

  my_kd_tree_2d_t index(2, pc, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  my_kd_tree_2d_t index_points(2, pc_points, nanoflann::KDTreeSingleIndexAdaptorParams(10));

  int time_step;
  float ret_float;
  float version;
  char ret_char[3];
  int old_exoid =
      ex_open(ExoFileOutMono, EX_READ, &exo->comp_wordsize, &exo->io_wordsize, &version);
  ex_inquire(old_exoid, EX_INQ_TIME, &time_step, &ret_float, ret_char);

  int num_nodal_vars;
  int err = ex_get_variable_param(old_exoid, EX_NODAL, &num_nodal_vars);
  CHECK_EX_ERROR(err, "ex_get_variable_param");

  std::vector<std::vector<double>> old_values_list(num_nodal_vars);
  for (int var = 0; var < num_nodal_vars; var++) {
    old_values_list[var].resize(exo->num_nodes);
    err = ex_get_var(old_exoid, time_step, EX_NODAL, var + 1, 1, old_values_list[var].size(),
                     old_values_list[var].data());
  }
  ex_close(old_exoid);

  std::vector<std::vector<double>> interpolated_values_list(num_nodal_vars);
  for (int var = 0; var < num_nodal_vars; var++) {
    interpolated_values_list[var].resize(num_nodes);
  }

  for (int i = 0; i < num_nodes; i++) {
    double query_pt[2] = {new_nodes[i * 3], new_nodes[i * 3 + 1]};

    const int num_results = 12;
    std::vector<size_t> ret_index(num_results);
    std::vector<double> out_dist_sqr(num_results);
    nanoflann::KNNResultSet<double> resultSet(num_results);
    resultSet.init(&ret_index[0], &out_dist_sqr[0]);

    index.findNeighbors(resultSet, query_pt, nanoflann::SearchParameters());
    bool found = false;
    dbl w1, w2, w3;
    int node_ids[4];
    for (size_t j = 0; j < resultSet.size(); j++) {
      int block_id = elem_to_block[ret_index[j]].first;
      int elem_id = elem_to_block[ret_index[j]].second;
      int node1_id = exo->eb_conn[block_id][elem_id * 3];
      int node2_id = exo->eb_conn[block_id][elem_id * 3 + 1];
      int node3_id = exo->eb_conn[block_id][elem_id * 3 + 2];

      found = triangle_linear_interp(query_pt[0], query_pt[1], exo->x_coord[node1_id],
                                     exo->y_coord[node1_id], exo->x_coord[node2_id],
                                     exo->y_coord[node2_id], exo->x_coord[node3_id],
                                     exo->y_coord[node3_id], w1, w2, w3);

      if (found) {
        node_ids[0] = node1_id;
        node_ids[1] = node2_id;
        node_ids[2] = node3_id;
        break;
      }
    }

    if (!found) {
      const int nearest_results = 1;
      std::vector<size_t> nearest_index(nearest_results);
      std::vector<double> nearest_dist_sqr(nearest_results);
      nanoflann::KNNResultSet<double> nearestSet(nearest_results);
      nearestSet.init(&nearest_index[0], &nearest_dist_sqr[0]);
      index_points.findNeighbors(nearestSet, query_pt, nanoflann::SearchParameters());
      int nearest_node_id = nearest_index[0];
      for (int var = 0; var < num_nodal_vars; var++) {
        interpolated_values_list[var][i] = old_values_list[var][nearest_node_id];
      }
    } else {
      for (int var = 0; var < num_nodal_vars; var++) {
        interpolated_values_list[var][i] = w1 * old_values_list[var][node_ids[0]] +
                                           w2 * old_values_list[var][node_ids[1]] +
                                           w3 * old_values_list[var][node_ids[2]];
      }
    }
  }

  for (int var = 0; var < num_nodal_vars; var++) {
    int status =
        ex_put_var(exo->exoid, 1, EX_NODAL, var + 1, 1, interpolated_values_list[var].size(),
                   interpolated_values_list[var].data());
    GOMA_EH(status, "ex_put_var");
  }
}
#endif

void interp_solution_to_new_mesh_3d(Exo_DB *exo,
                                    Dpi *dpi,
                                    struct Results_Description **rd,
                                    double **x,
                                    double **xdot,
                                    double time1,
                                    double theta,
                                    double delta_t,
                                    double *new_nodes,
                                    int num_nodes) {
  std::vector<std::array<double, 3>> elements_centroids;
  std::vector<std::pair<int, int>> elem_to_block;
  for (int i = 0; i < exo->num_elem_blocks; i++) {
    for (int j = 0; j < exo->eb_num_elems[i]; j++) {
      elem_to_block.push_back(std::make_pair(i, j));
      int node1_id = exo->eb_conn[i][j * 4];
      int node2_id = exo->eb_conn[i][j * 4 + 1];
      int node3_id = exo->eb_conn[i][j * 4 + 2];
      int node4_id = exo->eb_conn[i][j * 4 + 3];
      double x1 = exo->x_coord[node1_id];
      double y1 = exo->y_coord[node1_id];
      double z1 = exo->z_coord[node1_id];
      double x2 = exo->x_coord[node2_id];
      double y2 = exo->y_coord[node2_id];
      double z2 = exo->z_coord[node2_id];
      double x3 = exo->x_coord[node3_id];
      double y3 = exo->y_coord[node3_id];
      double z3 = exo->z_coord[node3_id];
      double x4 = exo->x_coord[node4_id];
      double y4 = exo->y_coord[node4_id];
      double z4 = exo->z_coord[node4_id];
      elements_centroids.push_back(
          {(x1 + x2 + x3 + x4) / 4.0, (y1 + y2 + y3 + y4) / 4.0, (z1 + z2 + z3 + z4) / 4.0});
    }
  }

  PointCloud<3> pc;
  pc.pts = elements_centroids;

  PointCloud<3> pc_points;
  for (int i = 0; i < exo->base_mesh->num_nodes; i++) {
    pc_points.pts.push_back({exo->x_coord[i], exo->y_coord[i], exo->z_coord[i]});
  }

  using my_kd_tree_3d_t =
      nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud<3>>,
                                          PointCloud<3>, 3 /* dim */>;

  my_kd_tree_3d_t index(3, pc, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  my_kd_tree_3d_t index_points(3, pc_points, nanoflann::KDTreeSingleIndexAdaptorParams(10));

  int time_step;
  float ret_float;
  float version;
  char ret_char[3];
  int old_exoid =
      ex_open(ExoFileOutMono, EX_READ, &exo->comp_wordsize, &exo->io_wordsize, &version);
  ex_inquire(old_exoid, EX_INQ_TIME, &time_step, &ret_float, ret_char);

  int num_nodal_vars;
  int err = ex_get_variable_param(old_exoid, EX_NODAL, &num_nodal_vars);
  CHECK_EX_ERROR(err, "ex_get_variable_param");

  std::vector<std::vector<double>> old_values_list(num_nodal_vars);
  for (int var = 0; var < num_nodal_vars; var++) {
    old_values_list[var].resize(exo->num_nodes);
    err = ex_get_var(old_exoid, time_step, EX_NODAL, var + 1, 1, old_values_list[var].size(),
                     old_values_list[var].data());
  }
  ex_close(old_exoid);

  std::vector<std::vector<double>> interpolated_values_list(num_nodal_vars);
  for (int var = 0; var < num_nodal_vars; var++) {
    interpolated_values_list[var].resize(num_nodes);
  }

  for (int i = 0; i < num_nodes; i++) {
    double query_pt[3] = {new_nodes[i * 3], new_nodes[i * 3 + 1], new_nodes[i * 3 + 2]};

    const int num_results = 12;
    std::vector<size_t> ret_index(num_results);
    std::vector<double> out_dist_sqr(num_results);
    nanoflann::KNNResultSet<double> resultSet(num_results);
    resultSet.init(&ret_index[0], &out_dist_sqr[0]);

    index.findNeighbors(resultSet, query_pt, nanoflann::SearchParameters());
    bool found = false;
    dbl w1, w2, w3, w4;
    int node_ids[4];
    for (size_t j = 0; j < resultSet.size(); j++) {
      int block_id = elem_to_block[ret_index[j]].first;
      int elem_id = elem_to_block[ret_index[j]].second;
      int node1_id = exo->eb_conn[block_id][elem_id * 4];
      int node2_id = exo->eb_conn[block_id][elem_id * 4 + 1];
      int node3_id = exo->eb_conn[block_id][elem_id * 4 + 2];
      int node4_id = exo->eb_conn[block_id][elem_id * 4 + 3];

      found = tetrahedron_linear_interp(
          query_pt[0], query_pt[1], query_pt[2], exo->x_coord[node1_id], exo->y_coord[node1_id],
          exo->z_coord[node1_id], exo->x_coord[node2_id], exo->y_coord[node2_id],
          exo->z_coord[node2_id], exo->x_coord[node3_id], exo->y_coord[node3_id],
          exo->z_coord[node3_id], exo->x_coord[node4_id], exo->y_coord[node4_id],
          exo->z_coord[node4_id], w1, w2, w3, w4);

      if (found) {
        node_ids[0] = node1_id;
        node_ids[1] = node2_id;
        node_ids[2] = node3_id;
        node_ids[3] = node4_id;
        break;
      }
    }

    if (!found) {
      const int nearest_results = 1;
      std::vector<size_t> nearest_index(nearest_results);
      std::vector<double> nearest_dist_sqr(nearest_results);
      nanoflann::KNNResultSet<double> nearestSet(nearest_results);
      nearestSet.init(&nearest_index[0], &nearest_dist_sqr[0]);
      index_points.findNeighbors(nearestSet, query_pt, nanoflann::SearchParameters());
      int nearest_node_id = nearest_index[0];
      for (int var = 0; var < num_nodal_vars; var++) {
        interpolated_values_list[var][i] = old_values_list[var][nearest_node_id];
      }
    } else {
      for (int var = 0; var < num_nodal_vars; var++) {
        interpolated_values_list[var][i] =
            w1 * old_values_list[var][node_ids[0]] + w2 * old_values_list[var][node_ids[1]] +
            w3 * old_values_list[var][node_ids[2]] + w4 * old_values_list[var][node_ids[3]];
      }
    }
  }

  for (int var = 0; var < num_nodal_vars; var++) {
    int status =
        ex_put_var(exo->exoid, 1, EX_NODAL, var + 1, 1, interpolated_values_list[var].size(),
                   interpolated_values_list[var].data());
    GOMA_EH(status, "ex_put_var");
  }
}

void convert_mesh_to_mmg(Exo_DB *exo,
                         Dpi *dpi,
                         double **x,
                         double **xdot,
                         double time1,
                         double theta,
                         double delta_t,
                         MMG5_pMesh *mmgMesh) {
  int32_t *tria = NULL;
  int32_t *refs = NULL;

  tria = (int32_t *)malloc(sizeof(int32_t) * exo->num_elems * 3);
  refs = (int32_t *)malloc(sizeof(int32_t) * exo->num_elems);

  int offset = 0;
  for (int ebn = 0; ebn < exo->num_elem_blocks; ebn++) {

    /*First we must calculate the material-referenced element
     *number so as to be compatible with the ElemStorage struct
     */
    int type =
        get_type(exo->eb_elem_type[ebn], exo->eb_num_nodes_per_elem[ebn], exo->eb_num_attr[ebn]);

    if (type != LINEAR_TRI) {
      GOMA_EH(GOMA_ERROR,
              "Only linear triangles are supported in this version of the MMG adapter.");
    }

    for (int ielem = 0; ielem < exo->eb_num_elems[ebn]; ielem++) {
      refs[offset] = exo->eb_id[ebn];
      for (int j = 0; j < 3; j++) {
        int node_id = exo->eb_conn[ebn][ielem * 3 + j];
        tria[offset * 3 + j] = node_id + 1;
      }
      offset++;
    }
  }

  double *coords = (double *)malloc(sizeof(double) * exo->num_nodes * 2);
  int32_t *coord_refs = (int32_t *)malloc(sizeof(int32_t) * exo->num_nodes);
  for (int i = 0; i < exo->num_nodes; i++) {
    coord_refs[i] = 0; /* no reference for vertices */
    coords[i * 2] = exo->x_coord[i];
    coords[i * 2 + 1] = exo->y_coord[i];
  }

  int num_edges = 0;
  for (int ss = 0; ss < exo->num_side_sets; ss++) {
    num_edges += exo->ss_num_sides[ss];
  }
  int32_t *edge_conns = (int32_t *)malloc(sizeof(int32_t) * num_edges * 2);
  int32_t *edge_refs = (int32_t *)malloc(sizeof(int32_t) * num_edges);
  int edge_offset = 0;
  for (int ss = 0; ss < exo->num_side_sets; ss++) {
    for (int iside = 0; iside < exo->ss_num_sides[ss]; iside++) {
      edge_conns[edge_offset * 2] = exo->ss_node_list[ss][exo->ss_node_side_index[ss][iside]] + 1;
      edge_conns[edge_offset * 2 + 1] =
          exo->ss_node_list[ss][exo->ss_node_side_index[ss][iside] + 1] + 1;
      edge_refs[edge_offset] = exo->ss_id[ss];
      edge_offset++;
    }
  }

  MMG2D_Set_meshSize(*mmgMesh, exo->num_nodes, exo->num_elems, 0, num_edges);

  for (int ns = 0; ns < exo->num_node_sets; ns++) {
    if (exo->ns_num_nodes[ns] == 1) {
      int node_id = exo->ns_node_list[exo->ns_node_index[ns]];
      MMG2D_Set_requiredVertex(*mmgMesh, node_id + 1);
      coord_refs[node_id] = exo->ns_id[ns];
    }
  }
  MMG2D_Set_vertices(*mmgMesh, coords, coord_refs);
  MMG2D_Set_triangles(*mmgMesh, tria, refs);
  MMG2D_Set_edges(*mmgMesh, edge_conns, edge_refs);

  free(tria);
  free(refs);
  free(coords);
  free(coord_refs);
  free(edge_conns);
  free(edge_refs);
}

void convert_mesh_to_mmg_3d(Exo_DB *exo,
                            Dpi *dpi,
                            double **x,
                            double **xdot,
                            double time1,
                            double theta,
                            double delta_t,
                            MMG5_pMesh *mmgMesh) {
  int32_t *tet = NULL;
  int32_t *refs = NULL;

  tet = (int32_t *)malloc(sizeof(int32_t) * exo->num_elems * 4);
  refs = (int32_t *)malloc(sizeof(int32_t) * exo->num_elems);

  int offset = 0;
  for (int ebn = 0; ebn < exo->num_elem_blocks; ebn++) {

    /*First we must calculate the material-referenced element
     *number so as to be compatible with the ElemStorage struct
     */
    int type =
        get_type(exo->eb_elem_type[ebn], exo->eb_num_nodes_per_elem[ebn], exo->eb_num_attr[ebn]);

    if (type != LINEAR_TET) {
      GOMA_EH(GOMA_ERROR,
              "Only linear tetrahedron are supported in this version of the MMG adapter.");
    }

    for (int ielem = 0; ielem < exo->eb_num_elems[ebn]; ielem++) {
      refs[offset] = exo->eb_id[ebn];
      for (int j = 0; j < 4; j++) {
        int node_id = exo->eb_conn[ebn][ielem * 4 + j];
        tet[offset * 4 + j] = node_id + 1;
      }
      offset++;
    }
  }

  double *coords = (double *)malloc(sizeof(double) * exo->num_nodes * 3);
  int32_t *coord_refs = (int32_t *)malloc(sizeof(int32_t) * exo->num_nodes);
  for (int i = 0; i < exo->num_nodes; i++) {
    coord_refs[i] = 0; /* no reference for vertices */
    coords[i * 3] = exo->x_coord[i];
    coords[i * 3 + 1] = exo->y_coord[i];
    coords[i * 3 + 2] = exo->z_coord[i];
  }

  int num_edges = 0;
  for (int ss = 0; ss < exo->num_side_sets; ss++) {
    num_edges += exo->ss_num_sides[ss];
  }
  int32_t *edge_conns = (int32_t *)malloc(sizeof(int32_t) * num_edges * 3);
  int32_t *edge_refs = (int32_t *)malloc(sizeof(int32_t) * num_edges);
  int edge_offset = 0;
  for (int ss = 0; ss < exo->num_side_sets; ss++) {
    for (int iside = 0; iside < exo->ss_num_sides[ss]; iside++) {
      edge_conns[edge_offset * 3] = exo->ss_node_list[ss][exo->ss_node_side_index[ss][iside]] + 1;
      edge_conns[edge_offset * 3 + 1] =
          exo->ss_node_list[ss][exo->ss_node_side_index[ss][iside] + 1] + 1;
      edge_conns[edge_offset * 3 + 2] =
          exo->ss_node_list[ss][exo->ss_node_side_index[ss][iside] + 2] + 1;
      edge_refs[edge_offset] = exo->ss_id[ss];
      edge_offset++;
    }
  }

  MMG3D_Set_meshSize(*mmgMesh, exo->num_nodes, exo->num_elems, 0, num_edges, 0, 0);

  for (int ns = 0; ns < exo->num_node_sets; ns++) {
    if (exo->ns_num_nodes[ns] == 1) {
      int node_id = exo->ns_node_list[exo->ns_node_index[ns]];
      MMG3D_Set_requiredVertex(*mmgMesh, node_id + 1);
      coord_refs[node_id] = exo->ns_id[ns];
    }
  }
  MMG3D_Set_vertices(*mmgMesh, coords, coord_refs);
  MMG3D_Set_tetrahedra(*mmgMesh, tet, refs);
  MMG3D_Set_triangles(*mmgMesh, edge_conns, edge_refs);

  free(tet);
  free(refs);
  free(coords);
  free(coord_refs);
  free(edge_conns);
  free(edge_refs);
}

void mmg_convert_to_exodus(MMG5_pMesh *mmgMesh,
                           struct Results_Description **rd,
                           Exo_DB *exo,
                           Dpi *dpi,
                           double **x,
                           double **xdot,
                           double time1,
                           double theta,
                           double delta_t) {
  int numVerticesNew, numCellsNew, numFacesNew;
  MMG2D_Get_meshSize(*mmgMesh, &numVerticesNew, &numCellsNew, 0, &numFacesNew);
  int32_t *verTagsNew, *corners, *requiredVer;
  double *verticesNew;
  verticesNew = (double *)malloc(sizeof(double) * 2 * numVerticesNew);
  verTagsNew = (int32_t *)malloc(sizeof(int32_t) * numVerticesNew);
  corners = (int32_t *)malloc(sizeof(int32_t) * numVerticesNew);
  requiredVer = (int32_t *)malloc(sizeof(int32_t) * numVerticesNew);
  int32_t *cellTagsNew, *requiredCells, *cellsNew;
  cellTagsNew = (int32_t *)malloc(sizeof(int32_t) * numCellsNew);
  requiredCells = (int32_t *)malloc(sizeof(int32_t) * numCellsNew);
  cellsNew = (int32_t *)malloc(sizeof(int32_t) * 3 * numCellsNew);
  int32_t *faceTagsNew, *ridges, *requiredFaces, *facesNew;
  faceTagsNew = (int32_t *)malloc(sizeof(int32_t) * numFacesNew);
  ridges = (int32_t *)malloc(sizeof(int32_t) * numFacesNew);
  requiredFaces = (int32_t *)malloc(sizeof(int32_t) * numFacesNew);
  facesNew = (int32_t *)malloc(sizeof(int32_t) * 2 * numFacesNew);

  MMG2D_Get_vertices(*mmgMesh, verticesNew, verTagsNew, corners, requiredVer);
  MMG2D_Get_triangles(*mmgMesh, cellsNew, cellTagsNew, requiredCells);
  MMG2D_Get_edges(*mmgMesh, facesNew, faceTagsNew, ridges, requiredFaces);

  /* Here you would convert the MMG mesh back to Exodus format and populate the Exo_DB struct */
  /* This is a placeholder and would need to be implemented based on your specific requirements */

  // ex_create("adapted", EXCLOBBER, comp_ws, io_ws)

  exo->cmode = EX_CLOBBER;
  exo->exoid = ex_create("tmp.mmg_adapted.e", exo->cmode, &exo->comp_wordsize, &exo->io_wordsize);

  int status = ex_put_init(exo->exoid, exo->title, exo->num_dim, numVerticesNew, numCellsNew,
                           exo->num_elem_blocks, exo->num_node_sets, exo->num_side_sets);

  double *x_coord = (double *)malloc(sizeof(double) * numVerticesNew);
  double *y_coord = (double *)malloc(sizeof(double) * numVerticesNew);
  for (int i = 0; i < numVerticesNew; i++) {
    x_coord[i] = verticesNew[i * 2];
    y_coord[i] = verticesNew[i * 2 + 1];
  }
  status = ex_put_coord(exo->exoid, x_coord, y_coord, NULL);
  GOMA_EH(status, "ex_put_coord");

  /*
   * ELEMENT BLOCKS...
   */
  Exo_DB exo_base_s;
  Exo_DB *exo_base = &exo_base_s;
  exo_base->num_elem_blocks = exo->num_elem_blocks;
  exo_base->eb_id = (int *)malloc(sizeof(int) * exo_base->num_elem_blocks);
  exo_base->eb_elem_type = (char **)malloc(sizeof(char *) * exo_base->num_elem_blocks);
  exo_base->eb_num_elems = (int *)malloc(sizeof(int) * exo_base->num_elem_blocks);

  std::map<std::vector<int>, std::vector<std::pair<int, int>>> edge_map;

  if (exo_base->num_elem_blocks > 0) {
    int ielem = 0;
    for (int i = 0; i < exo_base->num_elem_blocks; i++) {
      std::vector<int> cell_nodes;
      for (int j = 0; j < numCellsNew; j++) {
        if (cellTagsNew[j] == exo->eb_id[i]) {
          cell_nodes.push_back(cellsNew[j * 3]);
          cell_nodes.push_back(cellsNew[j * 3 + 1]);
          cell_nodes.push_back(cellsNew[j * 3 + 2]);
          std::vector<int> edge1 = {cellsNew[j * 3], cellsNew[j * 3 + 1]};
          std::sort(edge1.begin(), edge1.end());
          std::vector<int> edge2 = {cellsNew[j * 3 + 1], cellsNew[j * 3 + 2]};
          std::sort(edge2.begin(), edge2.end());
          std::vector<int> edge3 = {cellsNew[j * 3], cellsNew[j * 3 + 2]};
          std::sort(edge3.begin(), edge3.end());
          edge_map[edge1].push_back(std::make_pair(1, ielem));
          edge_map[edge2].push_back(std::make_pair(2, ielem));
          edge_map[edge3].push_back(std::make_pair(3, ielem));
          ielem++;
        }
      }
      if (cell_nodes.size() > 0) {
        status = ex_put_block(exo->exoid, EX_ELEM_BLOCK, exo->eb_id[i], exo->eb_elem_type[i],
                              cell_nodes.size() / 3, exo->eb_num_nodes_per_elem[i], 0, 0,
                              exo->eb_num_attr[i]);
        GOMA_EH(status, "ex_put_blocks elem");

        status = ex_put_conn(exo->exoid, EX_ELEM_BLOCK, exo->eb_id[i], cell_nodes.data(), 0, 0);
        GOMA_EH(status, "ex_put_conn elem");
      }
    }
  }

  for (int ss = 0; ss < exo->num_side_sets; ss++) {
    std::vector<int> edge_elem;
    std::vector<int> edge_side;
    std::unordered_set<int> node_set;
    int id = exo->ss_id[ss];
    for (int i = 0; i < numFacesNew; i++) {
      if (faceTagsNew[i] == id) {
        std::vector<int> edge_key = {facesNew[i * 2], facesNew[i * 2 + 1]};
        std::sort(edge_key.begin(), edge_key.end());
        if (edge_map.find(edge_key) != edge_map.end()) {
          auto cells = edge_map[edge_key];
          for (auto cell : cells) {
            edge_side.push_back(cell.first);
            edge_elem.push_back(cell.second + 1);
          }
        } else {
          GOMA_EH(GOMA_ERROR, "Edge not found in edge map");
        }
        node_set.insert(facesNew[i * 2]);
        node_set.insert(facesNew[i * 2 + 1]);
      }
    }

    std::vector<int> node_set_vector(node_set.begin(), node_set.end());
    std::vector<double> node_set_dist(node_set_vector.size());
    std::fill(node_set_dist.begin(), node_set_dist.end(), 0.0);

    std::vector<double> side_set_dist(edge_side.size());
    std::fill(side_set_dist.begin(), side_set_dist.end(), 0.0);

    status = ex_put_set_param(exo->exoid, EX_NODE_SET, exo->ss_id[ss], node_set_vector.size(),
                              node_set_vector.size());
    GOMA_EH(status, "ex_put_set_param node set");
    status = ex_put_set_param(exo->exoid, EX_SIDE_SET, exo->ss_id[ss], edge_elem.size(),
                              edge_side.size());
    GOMA_EH(status, "ex_put_set_param side set");
    status = ex_put_set(exo->exoid, EX_NODE_SET, exo->ss_id[ss], node_set_vector.data(), NULL);
    GOMA_EH(status, "ex_put_set node set");
    status =
        ex_put_set(exo->exoid, EX_SIDE_SET, exo->ss_id[ss], edge_elem.data(), edge_side.data());
    GOMA_EH(status, "ex_put_set side set");
    status = ex_put_set_dist_fact(exo->exoid, EX_NODE_SET, exo->ss_id[ss], node_set_dist.data());
    GOMA_EH(status, "ex_put_set_dist_fact node set");
    status = ex_put_set_dist_fact(exo->exoid, EX_SIDE_SET, exo->ss_id[ss], side_set_dist.data());
    GOMA_EH(status, "ex_put_set_dist_fact side set");
  }
  for (int ns = 0; ns < exo->num_node_sets; ns++) {
    std::vector<int> node_set;
    int ns_id = exo->ns_id[ns];
    for (int i = 0; i < numVerticesNew; i++) {
      if (verTagsNew[i] == ns_id) {
        node_set.push_back(i + 1);
      }
    }
    if (node_set.size() == 1) {
      printf("NODE SET %d is a single node, setting it as required\n", ns_id);
      printf("NODE SET %d is a single node, setting it as required\n", ns_id);
      printf("NODE SET %d is a single node, setting it as required\n", ns_id);
      printf("NODE SET %d is a single node, setting it as required\n", ns_id);
      printf("NODE SET %d is a single node, setting it as required\n", ns_id);
      printf("NODE SET %d is a single node, setting it as required\n", ns_id);
      printf("NODE SET %d is a single node, setting it as required\n", ns_id);
      printf("NODE SET %d is a single node, setting it as required\n", ns_id);
      printf("NODE SET %d is a single node, setting it as required\n", ns_id);
      printf("NODE SET %d is a single node, setting it as required\n", ns_id);
      printf("NODE SET %d is a single node, setting it as required\n", ns_id);
      printf("NODE SET %d is a single node, setting it as required\n", ns_id);
      printf("NODE SET %d is a single node, setting it as required\n", ns_id);
      printf("NODE SET %d is a single node, setting it as required\n", ns_id);
      std::vector<double> node_set_dist(node_set.size());
      std::fill(node_set_dist.begin(), node_set_dist.end(), 0.0);
      status = ex_put_set_param(exo->exoid, EX_NODE_SET, exo->ns_id[ns], node_set.size(),
                                node_set.size());
      GOMA_EH(status, "ex_put_set_param node set");
      status = ex_put_set(exo->exoid, EX_NODE_SET, exo->ns_id[ns], node_set.data(), NULL);
      GOMA_EH(status, "ex_put_set node set");
      status = ex_put_set_dist_fact(exo->exoid, EX_NODE_SET, exo->ns_id[ns],
      node_set_dist.data()); GOMA_EH(status, "ex_put_set_dist_fact node set");
    }
  }

  char *var_names[MAX_NNV];
  int num_vars = 0;
  for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
    for (int i = 0; i < rd[imtrx]->nnv; i++) {
      var_names[num_vars + i] = rd[imtrx]->nvname[i];
    }
    num_vars += rd[imtrx]->nnv;
  }
  status = ex_put_variable_param(exo->exoid, EX_NODAL, num_vars);
  GOMA_EH(status, "ex_put_variable_param EX_NODAL");
  status = ex_put_variable_names(exo->exoid, EX_NODAL, num_vars, var_names);
  GOMA_EH(status, "ex_put_variable_names nodal");

  interp_solution_to_new_mesh_2d(exo, dpi, rd, x, xdot, time1, theta, delta_t, verticesNew,
                                 numVerticesNew);

  status = ex_close(exo->exoid);
  GOMA_EH(status, "ex_close");

  free(verticesNew);
  free(verTagsNew);
  free(corners);
  free(requiredVer);
  free(cellTagsNew);
  free(requiredCells);
  free(cellsNew);
  free(faceTagsNew);
  free(ridges);
  free(requiredFaces);
  free(facesNew);
  free(x_coord);
  free(y_coord);
  free(exo_base->eb_id);
  free(exo_base->eb_elem_type);
  free(exo_base->eb_num_elems);
}

void mmg_convert_to_exodus_3d(MMG5_pMesh *mmgMesh,
                              struct Results_Description **rd,
                              Exo_DB *exo,
                              Dpi *dpi,
                              double **x,
                              double **xdot,
                              double time1,
                              double theta,
                              double delta_t) {
  MMG5_int numVerticesNew = 0, numCellsNew = 0, numPrismsNew = 0, numFacesNew = 0;
  MMG5_int numQuadsNew = 0, numEdgesNew = 0;
  MMG3D_Get_meshSize(*mmgMesh, &numVerticesNew, &numCellsNew, &numPrismsNew, &numFacesNew,
                     &numQuadsNew, &numEdgesNew);

  MMG5_int *verTagsNew = (MMG5_int *)malloc(sizeof(MMG5_int) * numVerticesNew);
  int *corners = (int *)malloc(sizeof(int) * numVerticesNew);
  int *requiredVer = (int *)malloc(sizeof(int) * numVerticesNew);
  double *verticesNew = (double *)malloc(sizeof(double) * 3 * numVerticesNew);

  MMG5_int *cellTagsNew = (MMG5_int *)malloc(sizeof(MMG5_int) * numCellsNew);
  int *requiredCells = (int *)malloc(sizeof(int) * numCellsNew);
  MMG5_int *cellsNew = (MMG5_int *)malloc(sizeof(MMG5_int) * 4 * numCellsNew);

  MMG5_int *faceTagsNew = (MMG5_int *)malloc(sizeof(MMG5_int) * numFacesNew);
  int *requiredFaces = (int *)malloc(sizeof(int) * numFacesNew);
  MMG5_int *facesNew = (MMG5_int *)malloc(sizeof(MMG5_int) * 3 * numFacesNew);

  MMG3D_Get_vertices(*mmgMesh, verticesNew, verTagsNew, corners, requiredVer);
  MMG3D_Get_tetrahedra(*mmgMesh, cellsNew, cellTagsNew, requiredCells);
  MMG3D_Get_triangles(*mmgMesh, facesNew, faceTagsNew, requiredFaces);

  exo->cmode = EX_CLOBBER;
  exo->exoid = ex_create("tmp.mmg_adapted.e", exo->cmode, &exo->comp_wordsize, &exo->io_wordsize);

  int status = ex_put_init(exo->exoid, exo->title, exo->num_dim, numVerticesNew, numCellsNew,
                           exo->num_elem_blocks, exo->num_node_sets, exo->num_side_sets);

  double *x_coord = (double *)malloc(sizeof(double) * numVerticesNew);
  double *y_coord = (double *)malloc(sizeof(double) * numVerticesNew);
  double *z_coord = (double *)malloc(sizeof(double) * numVerticesNew);
  for (int i = 0; i < numVerticesNew; i++) {
    x_coord[i] = verticesNew[i * 3];
    y_coord[i] = verticesNew[i * 3 + 1];
    z_coord[i] = verticesNew[i * 3 + 2];
  }
  status = ex_put_coord(exo->exoid, x_coord, y_coord, z_coord);
  GOMA_EH(status, "ex_put_coord");

  std::map<std::vector<int>, std::vector<std::pair<int, int>>> face_map;
  int global_elem_id = 0;
  for (int i = 0; i < exo->num_elem_blocks; i++) {
    std::vector<int> cell_nodes;
    for (int j = 0; j < numCellsNew; j++) {
      if (cellTagsNew[j] == exo->eb_id[i]) {
        int v[4] = {(int)cellsNew[j * 4], (int)cellsNew[j * 4 + 1], (int)cellsNew[j * 4 + 2],
                    (int)cellsNew[j * 4 + 3]};

        cell_nodes.push_back(v[0]);
        cell_nodes.push_back(v[1]);
        cell_nodes.push_back(v[2]);
        cell_nodes.push_back(v[3]);

        for (int face = 0; face < 4; face++) {
          int local_indeces[MAX_NODES_PER_SIDE];
          sides2nodes(face, TETRAHEDRON, local_indeces);
          std::vector<int> face_key = {v[local_indeces[0]], v[local_indeces[1]],
                                       v[local_indeces[2]]};
          std::sort(face_key.begin(), face_key.end());
          face_map[face_key].push_back(std::make_pair(face + 1, global_elem_id));
        }
        global_elem_id++;
      }
    }

    if (cell_nodes.size() > 0) {
      status = ex_put_block(exo->exoid, EX_ELEM_BLOCK, exo->eb_id[i], exo->eb_elem_type[i],
                            cell_nodes.size() / 4, exo->eb_num_nodes_per_elem[i], 0, 0,
                            exo->eb_num_attr[i]);
      GOMA_EH(status, "ex_put_blocks elem");

      status = ex_put_conn(exo->exoid, EX_ELEM_BLOCK, exo->eb_id[i], cell_nodes.data(), 0, 0);
      GOMA_EH(status, "ex_put_conn elem");
    }
  }

  for (int ss = 0; ss < exo->num_side_sets; ss++) {
    std::vector<int> face_elem;
    std::vector<int> face_side;
    std::unordered_set<int> node_set;
    int id = exo->ss_id[ss];
    for (int i = 0; i < numFacesNew; i++) {
      if (faceTagsNew[i] == id) {
        std::vector<int> face_key = {(int)facesNew[i * 3], (int)facesNew[i * 3 + 1],
                                     (int)facesNew[i * 3 + 2]};
        std::sort(face_key.begin(), face_key.end());
        if (face_map.find(face_key) != face_map.end()) {
          auto cells = face_map[face_key];
          for (auto cell : cells) {
            face_side.push_back(cell.first);
            face_elem.push_back(cell.second + 1);
          }
        } else {
          GOMA_EH(GOMA_ERROR, "Face not found in face map");
        }
        node_set.insert(facesNew[i * 3]);
        node_set.insert(facesNew[i * 3 + 1]);
        node_set.insert(facesNew[i * 3 + 2]);
      }
    }

    std::vector<int> node_set_vector(node_set.begin(), node_set.end());
    std::vector<double> node_set_dist(node_set_vector.size());
    std::fill(node_set_dist.begin(), node_set_dist.end(), 0.0);

    std::vector<double> side_set_dist(face_side.size());
    std::fill(side_set_dist.begin(), side_set_dist.end(), 0.0);

    status = ex_put_set_param(exo->exoid, EX_NODE_SET, exo->ss_id[ss], node_set_vector.size(),
                              node_set_vector.size());
    GOMA_EH(status, "ex_put_set_param node set");
    status = ex_put_set_param(exo->exoid, EX_SIDE_SET, exo->ss_id[ss], face_elem.size(),
                              face_side.size());
    GOMA_EH(status, "ex_put_set_param side set");
    status = ex_put_set(exo->exoid, EX_NODE_SET, exo->ss_id[ss], node_set_vector.data(), NULL);
    GOMA_EH(status, "ex_put_set node set");
    status =
        ex_put_set(exo->exoid, EX_SIDE_SET, exo->ss_id[ss], face_elem.data(), face_side.data());
    GOMA_EH(status, "ex_put_set side set");
    status = ex_put_set_dist_fact(exo->exoid, EX_NODE_SET, exo->ss_id[ss], node_set_dist.data());
    GOMA_EH(status, "ex_put_set_dist_fact node set");
    status = ex_put_set_dist_fact(exo->exoid, EX_SIDE_SET, exo->ss_id[ss], side_set_dist.data());
    GOMA_EH(status, "ex_put_set_dist_fact side set");
  }

  for (int ns = 0; ns < exo->num_node_sets; ns++) {
    if (exo->ns_num_nodes[ns] == 1) {
      std::vector<int> node_set;
      int ns_id = exo->ns_id[ns];
      for (int i = 0; i < numVerticesNew; i++) {
        if (verTagsNew[i] == ns_id) {
          node_set.push_back(i + 1);
        }
      }
      if (node_set.size() == 1) {
        std::vector<double> node_set_dist(node_set.size());
        std::fill(node_set_dist.begin(), node_set_dist.end(), 0.0);
        status = ex_put_set_param(exo->exoid, EX_NODE_SET, exo->ns_id[ns], node_set.size(),
                                  node_set.size());
        GOMA_EH(status, "ex_put_set_param node set");
        status = ex_put_set(exo->exoid, EX_NODE_SET, exo->ns_id[ns], node_set.data(), NULL);
        GOMA_EH(status, "ex_put_set node set");
        status =
            ex_put_set_dist_fact(exo->exoid, EX_NODE_SET, exo->ns_id[ns], node_set_dist.data());
        GOMA_EH(status, "ex_put_set_dist_fact node set");
      }
    }
  }

  int offset = 0;
  char *var_names[MAX_NNV];
  for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
    int num_vars = rd[imtrx]->nnv;
    for (int i = 0; i < num_vars; i++) {
      var_names[offset + i] = rd[imtrx]->nvname[i];
    }
    offset += rd[imtrx]->nnv;
  }
  int num_vars = offset;

  status = ex_put_variable_param(exo->exoid, EX_NODAL, num_vars);
  GOMA_EH(status, "ex_put_variable_param EX_NODAL");
  status = ex_put_variable_names(exo->exoid, EX_NODAL, num_vars, var_names);
  GOMA_EH(status, "ex_put_variable_names nodal");

  interp_solution_to_new_mesh_3d(exo, dpi, rd, x, xdot, time1, theta, delta_t, verticesNew,
                                 numVerticesNew);

  status = ex_close(exo->exoid);
  GOMA_EH(status, "ex_close");

  free(verticesNew);
  free(verTagsNew);
  free(corners);
  free(requiredVer);
  free(cellTagsNew);
  free(requiredCells);
  free(cellsNew);
  free(faceTagsNew);
  free(requiredFaces);
  free(facesNew);
  free(x_coord);
  free(y_coord);
  free(z_coord);
}

void read_solution_exoII(Exo_DB *exo, Dpi *dpi, double **x, double **xdot) {
  char file[MAX_FNL] = "tmp.mmg_adapted.e";
  multiname(file, ProcID, Num_Proc);
  double time_value = 0.0;
  int msave = pg->imtrx;
  for (pg->imtrx = 0; pg->imtrx < upd->Total_Num_Matrices; pg->imtrx++) {
    int err = rd_vectors_from_exoII(x[pg->imtrx], file, 0, 0, INT_MAX, &time_value, exo);
    if (err != 0) {
      DPRINTF(stderr, "%s: err for rd_vectors_from_exoII()\n", "read_solution_exoII");
      exit(-1);
    }
  }
  pg->imtrx = msave;
}

Exo_DB *collect_mesh(Exo_DB *exo,
                     Dpi *dpi,
                     int imtrx,
                     double **x,
                     double **xdot,
                     double time1,
                     double theta,
                     double delta_t) {
  Exo_DB *exo_central = (Exo_DB *)malloc(sizeof(Exo_DB));
  init_exo_struct(exo_central);
  exo_central->base_mesh = NULL;
  Dpi *dpi_central = (Dpi *)malloc(sizeof(Dpi));
  // Here you would implement the logic to gather the mesh data from all processes
  // and populate the exo_central struct. This is a placeholder and would need to be
  // implemented based on your specific requirements and parallel communication setup.
  fix_output();
  rd_exo(exo_central, ExoFileOutMono, 0,
         (EXODB_ACTION_RD_INIT + EXODB_ACTION_RD_MESH + EXODB_ACTION_RD_RES0));
  uni_dpi(dpi_central, exo);

  int proc = Num_Proc;
  Num_Proc = 1;
  zero_base(exo_central);
  setup_base_mesh(dpi_central, exo_central, 1);
  uni_dpi(dpi_central, exo_central);
  // setup_old_dpi(exo_central, dpi_central);
  // setup_old_exo(exo_central, dpi_central, 1);
  Num_Proc = proc;

  free(dpi_central);

  return exo_central;
}

extern "C" void adapt_mesh_with_mmg(Exo_DB *exo,
                                    Dpi *dpi,
                                    struct Results_Description **rd,
                                    struct GomaLinearSolverData **ams,
                                    double **x,
                                    double **x_old,
                                    double **x_older,
                                    double **x_oldest,
                                    double **x_update,
                                    double **xdot,
                                    double **xdot_old,
                                    double **resid_vector,
                                    double **scale,
                                    double time1,
                                    double theta,
                                    double delta_t,
                                    double ****gvec_elem) {
  MMG5_pMesh mmgMesh;
  MMG5_pSol mmgSol;
  MMG5_int k, np;
  int ier;
  // outname = (char *)malloc(sizeof(char) * 256);

  fprintf(stdout, "  -- TEST MMG2DLIB \n");

  /** ------------------------------ STEP   I -------------------------- */
  /** 1) Initialisation of mesh and sol structures */
  /* args of InitMesh:
   * MMG5_ARG_start: we start to give the args of a variadic func
   * MMG5_ARG_ppMesh: next arg will be a pointer over a MMG5_pMesh
   * &mmgMesh: pointer toward your MMG5_pMesh (that store your mesh)
   * MMG5_ARG_ppMet: next arg will be a pointer over a MMG5_pSol storing a metric
   * &mmgSol: pointer toward your MMG5_pSol (that store your metric) */

  Exo_DB *exo_central = NULL;
  MPI_Barrier(MPI_COMM_WORLD);

  if (ProcID == 0) {
    exo_central = exo;
    if (Num_Proc > 1) {
      exo_central = collect_mesh(exo, dpi, 0, x, xdot, time1, theta, delta_t);
    }

    mmgMesh = NULL;
    mmgSol = NULL;
    if (exo->num_dim == 2) {
      MMG2D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmgMesh, MMG5_ARG_ppMet, &mmgSol,
                      MMG5_ARG_end);
    } else {
      MMG3D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmgMesh, MMG5_ARG_ppMet, &mmgSol,
                      MMG5_ARG_end);
    }

    /** 2) Build mesh in MMG5 format */
    /** Two solutions: just use the MMG2D_loadMesh function that will read a .mesh(b)
        file formatted or manually set your mesh using the MMG2D_Set* functions */

    /** with MMG2D_loadMesh function */
    //   if (MMG2D_loadMesh(mmgMesh, filename) != 1)
    // exit(EXIT_FAILURE);
    /** c) give solutions values and positions */
    int vdex;
    int time_step = 1;
    float version;
    float ret_float;
    char ret_char[MAX_STR_LENGTH];
    int exoII_id =
        ex_open(ExoFileOutMono, EX_READ, &exo->comp_wordsize, &exo->io_wordsize, &version);
    ex_inquire(exoII_id, EX_INQ_TIME, &time_step, &ret_float, ret_char);

    int num_nodal_vars;
    int err = ex_get_variable_param(exoII_id, EX_NODAL, &num_nodal_vars);
    CHECK_EX_ERROR(err, "ex_get_variable_param");

    std::vector<char> nodal_var_names_vec(num_nodal_vars * (MAX_STR_LENGTH + 1));
    std::vector<char *> nodal_var_names_ptrs(num_nodal_vars);
    for (int i = 0; i < num_nodal_vars; i++) {
      nodal_var_names_ptrs[i] = &nodal_var_names_vec[i * (MAX_STR_LENGTH + 1)];
    }
    if (num_nodal_vars > 0) {
      int err =
          ex_get_variable_names(exoII_id, EX_NODAL, num_nodal_vars, nodal_var_names_ptrs.data());
      CHECK_EX_ERROR(err, "ex_get_variable_names");
    }
    np = exo_central->num_nodes;

    // Update mesh coordinates with displacements
    // check for displacments
    bool mesh_enabled = false;
    for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
      if (upd->ep[imtrx][R_MESH1] >= 0) {
        mesh_enabled = true;
      }
    }
    //
    if (mesh_enabled) {
      vdex = -1;
      int offset = 0;
      for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
        for (int i = 0; i < rd[imtrx]->nnv; i++) {
          if (rd[imtrx]->nvtype[i] == R_MESH1 &&
              strcmp(nodal_var_names_ptrs[i + offset], rd[imtrx]->nvname[i]) == 0) {
            vdex = i + offset;
            break;
          }
        }
        offset += rd[imtrx]->nnv;
      }
      if (vdex == -1) {
        GOMA_EH(GOMA_ERROR, "DMX variable not found in Goma");
      }
      std::vector<double> dx_values(np);
      err = ex_get_var(exoII_id, time_step, EX_NODAL, vdex + 1, 1, np, dx_values.data());
      CHECK_EX_ERROR(err, "ex_get_var");
      vdex = -1;
      for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
        for (int i = 0; i < rd[imtrx]->nnv; i++) {
          if (rd[imtrx]->nvtype[i] == R_MESH2 &&
              strcmp(nodal_var_names_ptrs[i + offset], rd[imtrx]->nvname[i]) == 0) {
            vdex = i + offset;
            break;
          }
        }
        offset += rd[imtrx]->nnv;
      }
      if (vdex == -1) {
        GOMA_EH(GOMA_ERROR, "DMY variable not found in Goma");
      }
      std::vector<double> dy_values(np);
      err = ex_get_var(exoII_id, time_step, EX_NODAL, vdex + 1, 1, np, dy_values.data());
      CHECK_EX_ERROR(err, "ex_get_var");

      std::vector<double> dz_values(np);
      if (exo->num_dim == 3) {
        vdex = -1;
        for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
          for (int i = 0; i < rd[imtrx]->nnv; i++) {
            if (rd[imtrx]->nvtype[i] == R_MESH3 &&
                strcmp(nodal_var_names_ptrs[i + offset], rd[imtrx]->nvname[i]) == 0) {
              vdex = i + offset;
              break;
            }
          }
          offset += rd[imtrx]->nnv;
        }
        if (vdex == -1) {
          GOMA_EH(GOMA_ERROR, "DMY variable not found in Goma");
        }
        err = ex_get_var(exoII_id, time_step, EX_NODAL, vdex + 1, 1, np, dz_values.data());
        CHECK_EX_ERROR(err, "ex_get_var");
      }
      for (int i = 0; i < np; i++) {
        exo_central->x_coord[i] += dx_values[i];
        exo_central->y_coord[i] += dy_values[i];
        if (exo->num_dim == 3) {
          exo_central->z_coord[i] += dz_values[i];
        }
      }
    }

    /** 3) Build sol in MMG5 format */
    /** Two solutions: just use the MMG2D_loadMet function that will read a .sol(b)
        file formatted or manually set your sol using the MMG2D_Set* functions */
    if (exo->num_dim == 2) {
      convert_mesh_to_mmg(exo_central, dpi, x, xdot, time1, theta, delta_t, &mmgMesh);

      /** Manually set of the sol */
      /** a) Get np the number of vertex */
      if (MMG2D_Get_meshSize(mmgMesh, &np, NULL, NULL, NULL) != 1)
        exit(EXIT_FAILURE);

      if (MMG2D_Set_solSize(mmgMesh, mmgSol, MMG5_Vertex, np, MMG5_Scalar) != 1)
        exit(EXIT_FAILURE);
    } else {
      convert_mesh_to_mmg_3d(exo_central, dpi, x, xdot, time1, theta, delta_t, &mmgMesh);

      /** Manually set of the sol */
      /** a) Get np the number of vertex */
      if (MMG3D_Get_meshSize(mmgMesh, &np, NULL, NULL, NULL, NULL, NULL) != 1)
        exit(EXIT_FAILURE);
      if (MMG3D_Set_solSize(mmgMesh, mmgSol, MMG5_Vertex, np, MMG5_Scalar) != 1)
        exit(EXIT_FAILURE);
    }

    /** b) give info for the sol structure: sol applied on vertex entities,
        number of vertices=np, the sol is scalar*/

    /** c) give solutions values and positions */

    vdex = -1;
    int offset = 0;
    for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
      for (int i = 0; i < rd[imtrx]->nnv; i++) {
        if (rd[imtrx]->nvtype[i] == ls->var &&
            strcmp(nodal_var_names_ptrs[i + offset], rd[imtrx]->nvname[i]) == 0) {
          vdex = i + offset;
          break;
        }
      }
      offset += rd[imtrx]->nnv;
    }
    if (vdex == -1) {
      GOMA_EH(GOMA_ERROR, "Level set variable not found in Goma");
    }
    std::vector<double> ls_values(np);
    err = ex_get_var(exoII_id, time_step, EX_NODAL, vdex + 1, 1, np, ls_values.data());
    CHECK_EX_ERROR(err, "ex_get_var");
    ex_close(exoII_id);

    for (k = 1; k <= np; k++) {
      double ls_value = ls->adapt_outer_size;
      if (fabs(ls_values[k - 1]) < ls->adapt_width) {
        ls_value = ls->adapt_inner_size;
      }

      if (exo->num_dim == 2) {
        if (MMG2D_Set_scalarSol(mmgSol, ls_value, k) != 1)
          exit(EXIT_FAILURE);
      } else {
        if (MMG3D_Set_scalarSol(mmgSol, ls_value, k) != 1)
          exit(EXIT_FAILURE);
      }
    }

    /** 4) (not mandatory): check if the number of given entities match with mesh size */
    if (exo->num_dim == 2) {
      if (MMG2D_Chk_meshData(mmgMesh, mmgSol) != 1)
        exit(EXIT_FAILURE);

      ier = MMG2D_mmg2dlib(mmgMesh, mmgSol);
    } else {
      if (MMG3D_Chk_meshData(mmgMesh, mmgSol) != 1)
        exit(EXIT_FAILURE);

      ier = MMG3D_mmg3dlib(mmgMesh, mmgSol);
    }

    if (ier == MMG5_STRONGFAILURE) {
      GOMA_EH(GOMA_ERROR, "BAD ENDING OF MMG2DLIB: UNABLE TO SAVE MESH\n");

    } else if (ier == MMG5_LOWFAILURE)
      GOMA_EH(GOMA_ERROR, "BAD ENDING OF MMG2DLIB\n");

    // snprintf(outname, 256, "adapted_mesh_%d.mesh", 1);
    /*save result*/
    // if (MMG2D_saveMesh(mmgMesh, outname) != 1)
    //   exit(EXIT_FAILURE);

    // /*save metric*/
    // if (MMG2D_saveSol(mmgMesh, mmgSol, outname) != 1)
    //   exit(EXIT_FAILURE);
    if (exo->num_dim == 2) {
      mmg_convert_to_exodus(&mmgMesh, rd, exo_central, dpi, x, xdot, time1, theta, delta_t);
      // GOMA_EH(GOMA_ERROR, "MMG2D -> EXODUS conversion not implemented for 2D mesh\n");
    } else {
      mmg_convert_to_exodus_3d(&mmgMesh, rd, exo_central, dpi, x, xdot, time1, theta, delta_t);
    }
  }

  for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
    free(idv[imtrx]);
  }
  free(idv);
  idv = NULL;
  if (rotation_allocated) {
    for (int i = 0; i < exo->num_nodes; i++) {
      for (int j = 0; j < NUM_VECTOR_EQUATIONS; j++) {
        free(rotation[i][j]);
      }
      free(rotation[i]);
      free(local_ROT_list[i]);
    }
    free(rotation);
    rotation = NULL;
    free(local_ROT_list);
    local_ROT_list = NULL;
    rotation_allocated = FALSE;
  }

  free_Surf_BC(First_Elem_Side_BC_Array, exo);
  free_Edge_BC(First_Elem_Edge_BC_Array, exo, dpi);
  free_nodes();

  if (goma_automatic_rotations.rotation_nodes != NULL) {
    for (int i = 0; i < exo->num_nodes; i++) {
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

  static bool first_call = true;
  static std::string base_name;
  static int step = 1;

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
  step++;

  //  const char * tmpfile = "";
  strncpy(ExoFile, "tmp.mmg_adapted.e", 127);
  strncpy(ExoFileOutMono, ss2.str().c_str(), 127);
  strncpy(ExoFileOut, ss2.str().c_str(), 127);
  multiname(ExoFileOut, ProcID, Num_Proc);
  int num_total_nodes = dpi->num_internal_nodes + dpi->num_boundary_nodes + dpi->num_external_nodes;

  for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
    for (int i = 0; i < num_total_nodes; i++) {
      free(Local_Offset[imtrx][i]);
      free(Dolphin[imtrx][i]);
    }
    free(Dolphin[imtrx]);
    free(Local_Offset[imtrx]);
  }
  safer_free((void **)&Local_Offset);
  safer_free((void **)&Dolphin);

  std::vector<char *> mmg_files;
  std::string file = "tmp.mmg_adapted.e";
  mmg_files.push_back(const_cast<char *>(file.c_str()));
  if (Num_Proc > 1 && ProcID == 0) {
    goma_metis_decomposition(mmg_files.data(), 1);
  }

  if (Num_Proc == 1) {
    free_dpi_uni(dpi);
  } else {
    free_dpi(dpi);
  }
  free_exo(exo);
  init_exo_struct(exo);
  init_dpi_struct(dpi);

  MPI_Barrier(MPI_COMM_WORLD);

  read_mesh_exoII(exo, dpi);
  one_base(exo, Num_Proc);
  wr_mesh_exo(exo, ExoFileOut, 0);
  zero_base(exo);

  if (upd->Total_Num_Matrices == 1) {
    wr_result_prelim_exo(rd[0], exo, ExoFileOut, gvec_elem[0]);
  } else {
    wr_result_prelim_exo_segregated(rd, exo, ExoFileOut, gvec_elem);
  }

  if (Num_Proc > 1) {
    wr_dpi(dpi, ExoFileOut);
  }
  if (dpi->num_neighbors > 0) {
    for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
      free(cx[imtrx]);
      free(Request);
      free(Status);
      cx[imtrx] = alloc_struct_1(Comm_Ex, DPI_ptr->num_neighbors);
      Request = alloc_struct_1(MPI_Request, Num_Requests * DPI_ptr->num_neighbors);
      Status = alloc_struct_1(MPI_Status, Num_Requests * DPI_ptr->num_neighbors);
    }
  }

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
    // realloc_dbl_1(&x_update[imtrx], numProcUnknowns + numProcUnknowns, 0);
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

  read_solution_exoII(exo, dpi, x, xdot);
  for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
    dcopy1(NumUnknowns[imtrx] + NumExtUnknowns[imtrx], x[imtrx], x_old[imtrx]);
    dcopy1(NumUnknowns[imtrx] + NumExtUnknowns[imtrx], x[imtrx], x_older[imtrx]);
    dcopy1(NumUnknowns[imtrx] + NumExtUnknowns[imtrx], x[imtrx], x_oldest[imtrx]);
  }

  /** 5) Free the MMG3D5 structures */
  if (ProcID == 0) {
    if (exo->num_dim == 2) {
      MMG2D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmgMesh, MMG5_ARG_ppMet, &mmgSol,
                     MMG5_ARG_end);
    } else {
      MMG3D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmgMesh, MMG5_ARG_ppMet, &mmgSol,
                     MMG5_ARG_end);
    }
  }
  if (Num_Proc > 1) {
    free(exo_central);
  }
}
