#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

extern "C" {
#include "cfd_dem_coupling.h"
#include "cfd_dem_util.h"
#include "el_elm.h"
}

using Catch::Approx;

static void map_point(int shape, const double coords[][3], const double xi[3], double x[3]) {
  double N[CFD_DEM_MAX_VERTS];
  cfd_dem_vertex_shape(shape, xi, N, NULL);
  x[0] = x[1] = x[2] = 0.0;
  for (int i = 0; i < cfd_dem_num_vertices(shape); i++)
    for (int k = 0; k < 3; k++)
      x[k] += N[i] * coords[i][k];
}

TEST_CASE("vertex shape functions are a partition of unity", "[cfd_dem][shape]") {
  const int shapes[] = {TRIANGLE, QUADRILATERAL, TETRAHEDRON, HEXAHEDRON};
  const double xi[3] = {0.21, 0.13, 0.17};
  for (int shape : shapes) {
    double N[CFD_DEM_MAX_VERTS], dN[CFD_DEM_MAX_VERTS][3];
    REQUIRE(cfd_dem_vertex_shape(shape, xi, N, dN) == 0);
    double sum = 0.0, dsum[3] = {0, 0, 0};
    for (int i = 0; i < cfd_dem_num_vertices(shape); i++) {
      sum += N[i];
      for (int k = 0; k < 3; k++)
        dsum[k] += dN[i][k];
    }
    CHECK(sum == Approx(1.0));
    for (int k = 0; k < 3; k++)
      CHECK(dsum[k] == Approx(0.0).margin(1e-14));
  }
}

TEST_CASE("triangle and tet vertex functions follow rf_shape.c ordering", "[cfd_dem][shape]") {
  double N[CFD_DEM_MAX_VERTS];
  const double xi[3] = {0.2, 0.3, 0.1};
  cfd_dem_vertex_shape(TRIANGLE, xi, N, NULL);
  CHECK(N[0] == Approx(0.2)); /* s */
  CHECK(N[1] == Approx(0.3)); /* t */
  CHECK(N[2] == Approx(0.5)); /* 1 - s - t */
  cfd_dem_vertex_shape(TETRAHEDRON, xi, N, NULL);
  CHECK(N[0] == Approx(0.4)); /* 1 - s - t - u */
  CHECK(N[1] == Approx(0.2));
  CHECK(N[2] == Approx(0.3));
  CHECK(N[3] == Approx(0.1));
}

TEST_CASE("inverse map recovers xi in distorted elements", "[cfd_dem][invert]") {
  SECTION("quad") {
    const double c[4][3] = {{0, 0, 0}, {2.0, 0.1, 0}, {2.3, 1.7, 0}, {-0.2, 1.2, 0}};
    const double xi_true[3] = {0.37, -0.61, 0.0};
    double x[3], xi[3];
    map_point(QUADRILATERAL, c, xi_true, x);
    cfd_dem_shape_center(QUADRILATERAL, xi);
    REQUIRE(cfd_dem_invert_map(QUADRILATERAL, c, x, xi) == 0);
    CHECK(xi[0] == Approx(xi_true[0]));
    CHECK(xi[1] == Approx(xi_true[1]));
    CHECK(cfd_dem_xi_inside(QUADRILATERAL, xi, 1e-10));
  }
  SECTION("hex") {
    double c[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                      {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    c[6][0] = 1.3, c[6][1] = 1.2, c[6][2] = 1.1; /* distort one corner */
    const double xi_true[3] = {-0.3, 0.8, 0.25};
    double x[3], xi[3];
    map_point(HEXAHEDRON, c, xi_true, x);
    cfd_dem_shape_center(HEXAHEDRON, xi);
    REQUIRE(cfd_dem_invert_map(HEXAHEDRON, c, x, xi) == 0);
    for (int k = 0; k < 3; k++)
      CHECK(xi[k] == Approx(xi_true[k]));
  }
  SECTION("triangle, point outside") {
    const double c[3][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    const double x[3] = {0.8, 0.8, 0.0};
    double xi[3];
    cfd_dem_shape_center(TRIANGLE, xi);
    REQUIRE(cfd_dem_invert_map(TRIANGLE, c, x, xi) == 0);
    CHECK_FALSE(cfd_dem_xi_inside(TRIANGLE, xi, 1e-10));
  }
}

TEST_CASE("lumped vertex volumes sum to the element volume", "[cfd_dem][volume]") {
  SECTION("unit square") {
    const double c[4][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    double vol[CFD_DEM_MAX_VERTS];
    CHECK(cfd_dem_vertex_volumes(QUADRILATERAL, c, vol) == Approx(1.0));
    for (int i = 0; i < 4; i++)
      CHECK(vol[i] == Approx(0.25));
  }
  SECTION("general quad matches shoelace area") {
    const double c[4][3] = {{0, 0, 0}, {2.0, 0.1, 0}, {2.3, 1.7, 0}, {-0.2, 1.2, 0}};
    double area = 0.0;
    for (int i = 0; i < 4; i++) {
      const int j = (i + 1) % 4;
      area += c[i][0] * c[j][1] - c[j][0] * c[i][1];
    }
    area = 0.5 * fabs(area);
    double vol[CFD_DEM_MAX_VERTS];
    const double total = cfd_dem_vertex_volumes(QUADRILATERAL, c, vol);
    CHECK(total == Approx(area));
    CHECK(vol[0] + vol[1] + vol[2] + vol[3] == Approx(area));
    for (int i = 0; i < 4; i++)
      CHECK(vol[i] > 0.0);
  }
  SECTION("tet") {
    const double c[4][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    double vol[CFD_DEM_MAX_VERTS];
    CHECK(cfd_dem_vertex_volumes(TETRAHEDRON, c, vol) == Approx(1.0 / 6.0));
    for (int i = 0; i < 4; i++)
      CHECK(vol[i] == Approx(1.0 / 24.0));
  }
}

TEST_CASE("Stokes drag", "[cfd_dem][drag]") {
  const double u[3] = {0.5, -0.25, 0.0};
  double F[3];
  REQUIRE(cfd_dem_drag_force(CFD_DEM_DRAG_STOKES, 1000.0, 1e-3, 0.7, 1e-4, u, 3, F) == 0);
  const double c = 3.0 * M_PI * 1e-3 * 1e-4;
  CHECK(F[0] == Approx(c * 0.5));
  CHECK(F[1] == Approx(c * -0.25));
  CHECK(F[2] == 0.0);
}

TEST_CASE("Di Felice drag matches the textbook form", "[cfd_dem][drag]") {
  const double rho = 1000.0, mu = 1e-3, d = 1e-3;
  const double eps_values[] = {1.0, 0.8, 0.5};
  const double speeds[] = {1e-3, 1e-2, 0.1, 1.0};
  for (double eps : eps_values) {
    for (double U : speeds) {
      const double u[3] = {U, 0.0, 0.0};
      double F[3];
      REQUIRE(cfd_dem_drag_force(CFD_DEM_DRAG_DI_FELICE, rho, mu, eps, d, u, 2, F) == 0);
      const double Re = rho * eps * d * U / mu;
      const double Cd = pow(0.63 + 4.8 / sqrt(Re), 2);
      const double chi = 3.7 - 0.65 * exp(-0.5 * pow(1.5 - log10(Re), 2));
      const double expected =
          0.5 * Cd * rho * M_PI * d * d / 4.0 * eps * eps * U * U * pow(eps, -chi);
      CHECK(F[0] == Approx(expected).epsilon(1e-12));
      CHECK(F[1] == 0.0);
    }
  }
}

TEST_CASE("Di Felice drag is regular as the slip velocity vanishes", "[cfd_dem][drag]") {
  const double rho = 1.0, mu = 1.0, d = 1.0, eps = 0.6;
  double F[3];
  const double zero[3] = {0, 0, 0};
  REQUIRE(cfd_dem_drag_force(CFD_DEM_DRAG_DI_FELICE, rho, mu, eps, d, zero, 3, F) == 0);
  CHECK(F[0] == 0.0);
  CHECK(std::isfinite(F[0]));

  /* F -> (4.8^2/24) 3 pi mu d eps^(1-3.7) U as Re -> 0 */
  const double U = 1e-10;
  const double u[3] = {U, 0, 0};
  cfd_dem_drag_force(CFD_DEM_DRAG_DI_FELICE, rho, mu, eps, d, u, 3, F);
  const double limit = (4.8 * 4.8 / 24.0) * 3.0 * M_PI * mu * d * pow(eps, 1.0 - 3.7) * U;
  CHECK(F[0] == Approx(limit).epsilon(1e-4));
}
