#include "adapt/adapt_mmg.h"
#include <algorithm>
#include <cstdint>
#include <map>
#include <mmg/mmg2d/libmmg2d.h>
#include <nanoflann.hpp>
#include <set>
#include <sstream>
#include <stdio.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
extern "C" {
#include "dp_types.h"
#include "dpi.h"
#include "el_elm.h"
#include "exo_conn.h"
#include "exo_struct.h"
#include "mm_as.h"
#include "mm_as_structs.h"
#include "mm_bc.h"
#include "mm_eh.h"
#include "mm_interface.h"
#include "mm_unknown_map.h"
#include "rd_mesh.h"
#include "rf_allo.h"
#include "rf_bc.h"
#include "rf_bc_const.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "rf_fill_const.h"
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
#include "mm_unknown_map.h"
#include "rd_dpi.h"
#include "rd_exo.h"
#include "rd_mesh.h"
#include "rf_node_const.h"
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

std::pair<bool, double> triangle_linear_interp(double x,
                                               double y,
                                               double x1,
                                               double y1,
                                               double v1,
                                               double x2,
                                               double y2,
                                               double v2,
                                               double x3,
                                               double y3,
                                               double v3) {
  double w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) /
              ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
  double w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) /
              ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
  double w3 = 1.0 - w1 - w2;
  if (w1 < 0 || w2 < 0 || w3 < 0) {
    return {false, 0.0};
  }
  return {true, w1 * v1 + w2 * v2 + w3 * v3};
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

void interp_solution_to_new_mesh(Exo_DB *exo,
                                 Dpi *dpi,
                                 struct Results_Description *rd,
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

  for (int k = 0; k < rd->TotalNVSolnOutput; k++) {
    std::vector<double> interpolated_values(num_nodes);
    int var = rd->nvtype[k];
    for (int i = 0; i < num_nodes; i++) {
      double query_pt[2] = {new_nodes[i * 2], new_nodes[i * 2 + 1]};

      const int num_results = 8;
      std::vector<size_t> ret_index(num_results);
      std::vector<double> out_dist_sqr(num_results);
      nanoflann::KNNResultSet<double> resultSet(num_results);
      resultSet.init(&ret_index[0], &out_dist_sqr[0]);

      index.findNeighbors(resultSet, query_pt, nanoflann::SearchParameters());
      double interp_value = 0.0;
      double total_dist = 0.0;
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
        auto [success, val] = triangle_linear_interp(
            query_pt[0], query_pt[1], x1, y1, x[pg->imtrx][Index_Solution(node1_id, var, 0, 0, -1, pg->imtrx)],
            x2, y2, x[pg->imtrx][Index_Solution(node2_id, var, 0, 0, -1, pg->imtrx)], x3, y3,
            x[pg->imtrx][Index_Solution(node3_id, var, 0, 0, -1, pg->imtrx)]);
        if (success) {
          found = true;
          interpolated_values[i] = val;
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
      interpolated_values[i] = x[pg->imtrx][Index_Solution(nearest_node_id, var, 0, 0, -1, pg->imtrx)];
      }
    }

    int status = ex_put_var(exo->exoid, 1, EX_NODAL, k + 1, 1, interpolated_values.size(),
                            interpolated_values.data());
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
      double interp_value = 0.0;
      double total_dist = 0.0;
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
        auto [success, val] = triangle_linear_interp(
            query_pt[0], query_pt[1], x1, y1, efv->ext_fld_ndl_val[k][node1_id],
            x2, y2, efv->ext_fld_ndl_val[k][node2_id], x3, y3,
            efv->ext_fld_ndl_val[k][node3_id]);
        if (success) {
          found = true;
          interpolated_values[i] = val;
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

void convert_mesh_to_mmg(Exo_DB *exo,
                         Dpi *dpi,
                         int imtrx,
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
    int type = exo->eb_elem_itype[ebn];
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
  int edges = 0;
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

void mmg_convert_to_exodus(MMG5_pMesh *mmgMesh,
                           struct Results_Description *rd,
                           Exo_DB *exo,
                           Dpi *dpi,
                           int imtrx,
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
  exo_base->num_elem_blocks = 1;
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
      std::vector<double> node_set_dist(node_set.size());
      std::fill(node_set_dist.begin(), node_set_dist.end(), 0.0);
      status = ex_put_set_param(exo->exoid, EX_NODE_SET, exo->ns_id[ns], node_set.size(),
                                node_set.size());
      GOMA_EH(status, "ex_put_set_param node set");
      status = ex_put_set(exo->exoid, EX_NODE_SET, exo->ns_id[ns], node_set.data(), NULL);
      GOMA_EH(status, "ex_put_set node set");
      status = ex_put_set_dist_fact(exo->exoid, EX_NODE_SET, exo->ns_id[ns], node_set_dist.data());
      GOMA_EH(status, "ex_put_set_dist_fact node set");
    }
  }
  int num_vars = rd->nnv;
  char *var_names[MAX_NNV];
  for (int i = 0; i < num_vars; i++) {
    var_names[i] = rd->nvname[i];
  }
  status = ex_put_variable_param(exo->exoid, EX_NODAL, num_vars);
  GOMA_EH(status, "ex_put_variable_param EX_NODAL");
  status = ex_put_variable_names(exo->exoid, EX_NODAL, num_vars, var_names);
  GOMA_EH(status, "ex_put_variable_names nodal");

  interp_solution_to_new_mesh(exo, dpi, rd, x, xdot, time1, theta, delta_t, verticesNew,
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
}




void read_solution_exoII(Exo_DB *exo, Dpi *dpi, double **x, double **xdot) {
  char *file = "tmp.mmg_adapted.e";
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

void adapt_mesh_with_mmg(Exo_DB *exo,
                         Dpi *dpi,
                         struct Results_Description *rd,
                         int imtrx,
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
 double ***gvec_elem,
                         bool mapvar) {

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

  mmgMesh = NULL;
  mmgSol = NULL;
  MMG2D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmgMesh, MMG5_ARG_ppMet, &mmgSol, MMG5_ARG_end);

  /** 2) Build mesh in MMG5 format */
  /** Two solutions: just use the MMG2D_loadMesh function that will read a .mesh(b)
      file formatted or manually set your mesh using the MMG2D_Set* functions */

  /** with MMG2D_loadMesh function */
  //   if (MMG2D_loadMesh(mmgMesh, filename) != 1)
  // exit(EXIT_FAILURE);

  /** 3) Build sol in MMG5 format */
  /** Two solutions: just use the MMG2D_loadMet function that will read a .sol(b)
      file formatted or manually set your sol using the MMG2D_Set* functions */
  convert_mesh_to_mmg(exo, dpi, imtrx, x, xdot, time1, theta, delta_t, &mmgMesh);

  /** Manually set of the sol */
  /** a) Get np the number of vertex */
  if (MMG2D_Get_meshSize(mmgMesh, &np, NULL, NULL, NULL) != 1)
    exit(EXIT_FAILURE);

  /** b) give info for the sol structure: sol applied on vertex entities,
      number of vertices=np, the sol is scalar*/
  if (MMG2D_Set_solSize(mmgMesh, mmgSol, MMG5_Vertex, np, MMG5_Scalar) != 1)
    exit(EXIT_FAILURE);

  /** c) give solutions values and positions */
  int fill_matrix = upd->matrix_index[ls->var];
  for (k = 1; k <= np; k++) {
    int j = Index_Solution(k - 1, ls->var, 0, 0, -1, upd->matrix_index[ls->var]);
    double ls_value = ls->adapt_outer_size;
    if (fabs(x[fill_matrix][j]) < ls->adapt_width) {
      ls_value = ls->adapt_inner_size;
    }

    if (MMG2D_Set_scalarSol(mmgSol, ls_value, k) != 1)
      exit(EXIT_FAILURE);
  }

  /** 4) (not mandatory): check if the number of given entities match with mesh size */
  if (MMG2D_Chk_meshData(mmgMesh, mmgSol) != 1)
    exit(EXIT_FAILURE);

  ier = MMG2D_mmg2dlib(mmgMesh, mmgSol);

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
  mmg_convert_to_exodus(&mmgMesh, rd, exo, dpi, imtrx, x, xdot, time1, theta, delta_t);

  //  goma_metis_decomposition("adapted.e", Num_Proc);

  for (int imtrx = 0; imtrx < upd->Total_Num_Matrices; imtrx++) {
    free(idv[imtrx]);
  }
  free(idv);
  idv = NULL;

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
  static int step = 0;

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


  read_mesh_exoII(exo, dpi);
  one_base(exo, Num_Proc);
  wr_mesh_exo(exo, ExoFileOut, 0);
  zero_base(exo);

  wr_result_prelim_exo(rd, exo, ExoFileOut, gvec_elem);

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

  /** 5) Free the MMG3D5 structures */
  MMG2D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmgMesh, MMG5_ARG_ppMet, &mmgSol, MMG5_ARG_end);

  if (mapvar) {

    FILE *pfile;
    const char *data_to_pass = "run";

    // Open a pipe to the 'cat' command for writing (w)
    // The 'cat' command will read from its stdin (our pipe) and write to its stdout (our stdout by
    // default)
    pfile = popen(
        "mapvar -output mapvar.out -plot out.exoII -mesh adapted.e -interpolated adapted-interp.e",
        "w");

    if (pfile == NULL) {
      perror("popen failed");
      GOMA_EH(GOMA_ERROR, "Failed to open pipe to mapvar command");
    }

    // Write the data to the command's stdin via the file pointer
    fwrite(data_to_pass, 1, strlen(data_to_pass), pfile);

    // Close the pipe
    pclose(pfile);
  }
}