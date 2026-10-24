#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <vector>

extern "C" {
#include "dp_types.h"
#include "dpi.h"
#include "el_elm.h"
#include "exo_struct.h"
#include "mm_as_structs.h"

extern struct External_Field_Variables *efv;
double import_elem_size(const Exo_DB *exo, int el);
void set_import_field(int w, const double *values, int n);
int load_import_field(int w, Exo_DB *exo, Dpi *dpi, Comm_Ex *cx);
void rf_util_free_globals(void);
}

// A mesh holding one element of type elem_type with the given node coordinates
struct one_elem_mesh {
  Exo_DB exo{};
  std::vector<int> elem_node_pntr;
  std::vector<int> elem_node_list;
  int eb_ptr[2] = {0, 1};
  int eb_elem_itype[1];

  one_elem_mesh(int dim,
                int elem_type,
                std::vector<double> &x,
                std::vector<double> &y,
                std::vector<double> *z = nullptr) {
    int num_nodes = static_cast<int>(x.size());
    elem_node_pntr = {0, num_nodes};
    for (int i = 0; i < num_nodes; i++) {
      elem_node_list.push_back(i);
    }
    eb_elem_itype[0] = elem_type;
    exo.num_dim = dim;
    exo.num_nodes = num_nodes;
    exo.num_elems = 1;
    exo.num_elem_blocks = 1;
    exo.eb_ptr = eb_ptr;
    exo.eb_elem_itype = eb_elem_itype;
    exo.elem_node_pntr = elem_node_pntr.data();
    exo.elem_node_list = elem_node_list.data();
    exo.x_coord = x.data();
    exo.y_coord = y.data();
    exo.z_coord = z != nullptr ? z->data() : nullptr;
  }
};

TEST_CASE("import_elem_size quadrilateral", "[rf_util][import_elem_size]") {
  // 2 x 0.5 rectangle, rotated 90 degrees
  std::vector<double> x = {0.0, 0.0, -0.5, -0.5};
  std::vector<double> y = {0.0, 2.0, 2.0, 0.0};
  one_elem_mesh mesh(2, BILINEAR_QUAD, x, y);
  REQUIRE(import_elem_size(&mesh.exo, 0) == Catch::Approx(1.0));

  // nodes 4-8 of a biquadratic quad do not change the size
  std::vector<double> x9 = {0.0, 3.0, 3.0, 0.0, 1.5, 3.0, 1.5, 0.0, 1.5};
  std::vector<double> y9 = {0.0, 0.0, 2.0, 2.0, 0.0, 1.0, 2.0, 1.0, 1.0};
  one_elem_mesh mesh9(2, BIQUAD_QUAD, x9, y9);
  REQUIRE(import_elem_size(&mesh9.exo, 0) == Catch::Approx(6.0));
}

TEST_CASE("import_elem_size hexahedron", "[rf_util][import_elem_size]") {
  // 1 x 2 x 3 box
  std::vector<double> x = {0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0};
  std::vector<double> y = {0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0};
  std::vector<double> z = {0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0};
  one_elem_mesh mesh(3, TRILINEAR_HEX, x, y, &z);
  REQUIRE(import_elem_size(&mesh.exo, 0) == Catch::Approx(6.0));
}

TEST_CASE("import_elem_size triangle", "[rf_util][import_elem_size]") {
  // square of the length of the edge between nodes 0 and 1, as h_elem_siz() uses
  std::vector<double> x = {0.0, 3.0, 0.0};
  std::vector<double> y = {0.0, 4.0, 1.0};
  one_elem_mesh mesh(2, LINEAR_TRI, x, y);
  REQUIRE(import_elem_size(&mesh.exo, 0) == Catch::Approx(25.0));
}

TEST_CASE("load_import_field element values to nodes", "[rf_util][import_elem_size]") {
  // 1 x 1 and 2 x 1 quads side by side, sharing nodes 1 and 4
  //   3---4-------5
  //   | 0 |   1   |
  //   0---1-------2
  std::vector<double> x = {0.0, 1.0, 3.0, 0.0, 1.0, 3.0};
  std::vector<double> y = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
  std::vector<int> elem_node_pntr = {0, 4, 8};
  std::vector<int> elem_node_list = {0, 1, 4, 3, 1, 2, 5, 4};
  std::vector<int> node_elem_pntr = {0, 1, 3, 4, 5, 7, 8};
  std::vector<int> node_elem_list = {0, 0, 1, 1, 0, 0, 1, 1};
  int eb_ptr[2] = {0, 2};
  int eb_elem_itype[1] = {BILINEAR_QUAD};
  Exo_DB exo{};
  exo.num_dim = 2;
  exo.num_nodes = 6;
  exo.num_elems = 2;
  exo.num_elem_blocks = 1;
  exo.eb_ptr = eb_ptr;
  exo.eb_elem_itype = eb_elem_itype;
  exo.elem_node_pntr = elem_node_pntr.data();
  exo.elem_node_list = elem_node_list.data();
  exo.node_elem_pntr = node_elem_pntr.data();
  exo.node_elem_list = node_elem_list.data();
  exo.x_coord = x.data();
  exo.y_coord = y.data();
  Dpi dpi{}; // serial, no neighbors to exchange with

  struct External_Field_Variables *saved_efv = efv;
  auto *field = new External_Field_Variables{};
  efv = field;
  efv->Num_external_field = 1;
  strcpy(efv->name[0], "F");
  strcpy(efv->file_nm[0], "IMPORT_EV");

  std::vector<double> values = {1.0, 4.0};
  set_import_field(0, values.data(), exo.num_elems);
  REQUIRE(load_import_field(0, &exo, &dpi, nullptr) == 0);
  // shared nodes: the average weighted by element size, (1 * 1 + 4 * 2) / 3
  std::vector<double> expected = {1.0, 3.0, 4.0, 1.0, 3.0, 4.0};
  for (int i = 0; i < exo.num_nodes; i++) {
    REQUIRE(efv->ext_fld_ndl_val[0][i] == Catch::Approx(expected[i]));
  }

  rf_util_free_globals();
  efv = saved_efv;
  delete field;
}
