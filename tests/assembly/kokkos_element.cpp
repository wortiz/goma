#include <Kokkos_Core.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mm_kokkos_element.hpp"
extern "C" {
#include "el_elm.h"
// Use the production legacy routines, without pulling their host globals into kernels.
double shape(double s, double t, double u, int type, int quantity, int node);
void find_stu(int quadrature, int type, double *s, double *t, double *u);
double Gq_weight(int quadrature, int type);
}

namespace {
using namespace goma::assembly;

void check_close(double actual,
                 double expected,
                 double relative = 1.e-10,
                 double absolute = 1.e-12) {
  INFO("actual = " << actual << ", expected = " << expected);
  CHECK(std::abs(actual - expected) <= absolute + relative * std::abs(expected));
}

template <int Dim> void check_reference_basis() {
  using Topology = Q2Q1<Dim>;
  const int q2type = Dim == 2 ? BIQUAD_QUAD : TRIQUAD_HEX;
  const int q1type = Dim == 2 ? BILINEAR_QUAD : TRILINEAR_HEX;
  for (int q = 0; q <= Topology::quadrature_points; ++q) {
    double xi[3] = {0.217, -0.43, 0.31};
    if (q < Topology::quadrature_points) {
      double legacy[3];
      find_stu(q, q2type, &legacy[0], &legacy[1], &legacy[2]);
      check_close(Topology::quadrature(q, xi), Gq_weight(q, q2type), 1.e-14);
      for (int d = 0; d < Dim; ++d) {
        check_close(xi[d], legacy[d], 1.e-14);
      }
    }
    for (int family = 0; family < 2; ++family) {
      const int nodes = family == 0 ? Topology::velocity_nodes : Topology::pressure_nodes;
      const int type = family == 0 ? q2type : q1type;
      double sum = 0.0;
      double gradient[Dim]{};
      for (int node = 0; node < nodes; ++node) {
        const auto basis = family == 0 ? Topology::template basis<2>(node, xi)
                                       : Topology::template basis<1>(node, xi);
        INFO("dimension = " << Dim << ", point = " << q << ", family = " << family
                            << ", node = " << node);
        check_close(basis.value, shape(xi[0], xi[1], Dim == 3 ? xi[2] : 0.0, type, PSI, node));
        sum += basis.value;
        for (int d = 0; d < Dim; ++d) {
          check_close(basis.gradient[d],
                      shape(xi[0], xi[1], Dim == 3 ? xi[2] : 0.0, type, DPSI_S + d, node));
          gradient[d] += basis.gradient[d];
        }
      }
      check_close(sum, 1.0);
      for (int d = 0; d < Dim; ++d) {
        check_close(gradient[d], 0.0);
      }
    }
  }
  for (int node = 0; node < Topology::velocity_nodes; ++node) {
    double xi[Dim];
    for (int d = 0; d < Dim; ++d) {
      xi[d] = Topology::node_coordinate(node, d);
    }
    for (int i = 0; i < Topology::velocity_nodes; ++i) {
      check_close(Topology::template basis<2>(i, xi).value, i == node ? 1.0 : 0.0);
    }
  }
  for (int degree = 0; degree <= 5; ++degree) {
    for (int d = 0; d < Dim; ++d) {
      double integral = 0.0;
      for (int q = 0; q < Topology::quadrature_points; ++q) {
        double xi[Dim];
        const double weight = Topology::quadrature(q, xi);
        integral += weight * std::pow(xi[d], degree);
      }
      const double expected = degree % 2 == 1 ? 0.0 : (1 << Dim) / double(degree + 1);
      check_close(integral, expected);
    }
  }
}

template <int Dim> double affine_matrix(int d, int r) {
  return d == r ? 1.1 + 0.2 * d : 0.07 * (d + 1) * (r + 1);
}

template <int Dim> double mapped_coordinate(const double xi[Dim], int d, bool distorted) {
  double result = 1.7 + 0.3 * d;
  for (int r = 0; r < Dim; ++r) {
    result += affine_matrix<Dim>(d, r) * xi[r];
  }
  if (distorted) {
    result += (d + 1) * (0.035 * xi[0] * xi[1] + 0.02 * (1.0 - xi[0] * xi[0]));
    if constexpr (Dim == 3) {
      result += (d + 1) * 0.01 * xi[1] * xi[2];
    }
  }
  return result;
}

template <int Dim> double mapped_jacobian(const double xi[Dim], int d, int r, bool distorted) {
  double result = affine_matrix<Dim>(d, r);
  if (distorted) {
    if (r == 0) {
      result += (d + 1) * (0.035 * xi[1] - 0.04 * xi[0]);
    } else if (r == 1) {
      result += (d + 1) * 0.035 * xi[0];
      if constexpr (Dim == 3) {
        result += (d + 1) * 0.01 * xi[2];
      }
    } else {
      result += (d + 1) * 0.01 * xi[1];
    }
  }
  return result;
}

template <int Dim> ElementCoordinates coordinates(int elements = 2) {
  using Topology = Q2Q1<Dim>;
  ElementCoordinates result("test element coordinates", elements, Topology::velocity_nodes, Dim);
  auto host = Kokkos::create_mirror_view(result);
  for (int e = 0; e < elements; ++e) {
    for (int node = 0; node < Topology::velocity_nodes; ++node) {
      double xi[Dim];
      for (int d = 0; d < Dim; ++d) {
        xi[d] = Topology::node_coordinate(node, d);
      }
      for (int d = 0; d < Dim; ++d) {
        host(e, node, d) = mapped_coordinate<Dim>(xi, d, e % 2 == 1);
      }
    }
  }
  Kokkos::deep_copy(result, host);
  return result;
}

template <int Dim> void check_geometry() {
  using Topology = Q2Q1<Dim>;
  const auto x = coordinates<Dim>();
  const auto geometry = prepare_q2q1_geometry<Dim>(x);
  const auto status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), geometry.status);
  const auto points = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), geometry.points);
  for (int e = 0; e < 2; ++e) {
    REQUIRE(status(e) == 0);
    double volume = 0.0;
    for (int q = 0; q < Topology::quadrature_points; ++q) {
      const auto &point = points(e, q);
      double xi[Dim];
      Topology::quadrature(q, xi);
      volume += point.weight;
      for (int d = 0; d < Dim; ++d) {
        check_close(point.position[d], mapped_coordinate<Dim>(xi, d, e == 1));
      }
      for (int family = 0; family < 2; ++family) {
        const int nodes = family == 0 ? Topology::velocity_nodes : Topology::pressure_nodes;
        for (int node = 0; node < nodes; ++node) {
          const auto &basis = family == 0 ? point.velocity[node] : point.pressure[node];
          const auto reference = family == 0 ? Topology::template basis<2>(node, xi)
                                             : Topology::template basis<1>(node, xi);
          check_close(basis.value, reference.value);
          // Independently verify the chain rule with the analytic geometry Jacobian.
          for (int r = 0; r < Dim; ++r) {
            double derivative = 0.0;
            for (int d = 0; d < Dim; ++d) {
              derivative += basis.gradient[d] * mapped_jacobian<Dim>(xi, d, r, e == 1);
            }
            check_close(derivative, reference.gradient[r]);
          }
        }
      }
    }
    // The distortion is a rank-one perturbation with zero mean derivative.
    double determinant;
    if constexpr (Dim == 2) {
      determinant = affine_matrix<Dim>(0, 0) * affine_matrix<Dim>(1, 1) -
                    affine_matrix<Dim>(0, 1) * affine_matrix<Dim>(1, 0);
    } else {
      determinant = 0.0;
      for (int r = 0; r < 3; ++r) {
        determinant += affine_matrix<Dim>(0, r) *
                       (affine_matrix<Dim>(1, (r + 1) % 3) * affine_matrix<Dim>(2, (r + 2) % 3) -
                        affine_matrix<Dim>(1, (r + 2) % 3) * affine_matrix<Dim>(2, (r + 1) % 3));
      }
    }
    check_close(volume, (1 << Dim) * determinant);
  }
}

template <int Dim> struct ElementFixture {
  using Topology = Q2Q1<Dim>;
  ElementCoordinates x = coordinates<Dim>();
  ElementGeometry<Dim> geometry = prepare_q2q1_geometry<Dim>(x);
  ElementFields solution{"test solution", 2, Topology::dofs};
  ElementFields velocity_dot{"test velocity derivative", 2, Topology::velocity_dofs};
  Kokkos::View<ConstantFluid<Dim> *> materials{"test materials", 2};
  Kokkos::View<FlowExpressions<Dim> *> expressions{"test expressions", 2};
  double time_coefficient = 1.73;

  explicit ElementFixture(bool transient) {
    auto u = Kokkos::create_mirror_view(solution);
    auto dot = Kokkos::create_mirror_view(velocity_dot);
    auto material = Kokkos::create_mirror_view(materials);
    auto flow = Kokkos::create_mirror_view(expressions);
    const auto host_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
    for (int e = 0; e < 2; ++e) {
      material(e).density = 1.4 + 0.3 * e;
      material(e).viscosity = 0.27 + 0.1 * e;
      flow(e).transient = transient;
      flow(e).continuity = {true, 0.8};
      for (int a = 0; a < Dim; ++a) {
        material(e).source[a] = 0.6 - 0.2 * a;
        flow(e).momentum[a] = {
            {true, 0.9}, {true, 0.7 + 0.1 * a}, {true, 1.2 - 0.1 * a}, {true, 0.6}};
        for (int node = 0; node < Topology::velocity_nodes; ++node) {
          double value = 0.4 - 0.2 * a;
          for (int d = 0; d < Dim; ++d) {
            value += (0.13 + 0.07 * d - 0.11 * a) * host_x(e, node, d);
          }
          const int dof = a * Topology::velocity_nodes + node;
          u(e, dof) = value + 0.03 * std::sin(node + a);
          dot(e, dof) = -0.2 + 0.1 * a + 0.04 * node;
        }
      }
      for (int node = 0; node < Topology::pressure_nodes; ++node) {
        u(e, Topology::velocity_dofs + node) = 0.7 + 0.09 * node;
      }
    }
    Kokkos::deep_copy(solution, u);
    Kokkos::deep_copy(velocity_dot, dot);
    Kokkos::deep_copy(materials, material);
    Kokkos::deep_copy(expressions, flow);
  }

  ~ElementFixture() { Kokkos::fence(); }

  ElementResult<Dim> evaluate(EvaluationRequest request) const {
    return evaluate_q2q1_elements(geometry, solution, velocity_dot, materials, expressions,
                                  time_coefficient, request);
  }
};

template <int Dim> void check_element_derivatives(bool transient) {
  using Topology = Q2Q1<Dim>;
  ElementFixture<Dim> fixture(transient);
  const auto combined = fixture.evaluate(EvaluationRequest::both);
  const auto r_only = fixture.evaluate(EvaluationRequest::residual);
  const auto j_only = fixture.evaluate(EvaluationRequest::jacobian);
  REQUIRE(r_only.jacobian.size() == 0);
  REQUIRE(j_only.residual.size() == 0);
  const auto status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), combined.status);
  const auto residual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), combined.residual);
  const auto jacobian = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), combined.jacobian);
  const auto residual_only =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r_only.residual);
  const auto jacobian_only =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), j_only.jacobian);
  for (int e = 0; e < 2; ++e) {
    REQUIRE(status(e) == 0);
    for (int row = 0; row < Topology::dofs; ++row) {
      check_close(residual(e, row), residual_only(e, row));
      for (int column = 0; column < Topology::dofs; ++column) {
        check_close(jacobian(e, row, column), jacobian_only(e, row, column));
      }
    }
  }
  auto u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fixture.solution);
  auto dot = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fixture.velocity_dot);
  constexpr double step = 1.e-5;
  for (int column = 0; column < Topology::dofs; ++column) {
    for (int e = 0; e < 2; ++e) {
      u(e, column) += step;
      if (column < Topology::velocity_dofs) {
        dot(e, column) += step * fixture.time_coefficient;
      }
    }
    Kokkos::deep_copy(fixture.solution, u);
    Kokkos::deep_copy(fixture.velocity_dot, dot);
    const auto plus_result = fixture.evaluate(EvaluationRequest::residual);
    const auto plus =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), plus_result.residual);
    for (int e = 0; e < 2; ++e) {
      u(e, column) -= 2.0 * step;
      if (column < Topology::velocity_dofs) {
        dot(e, column) -= 2.0 * step * fixture.time_coefficient;
      }
    }
    Kokkos::deep_copy(fixture.solution, u);
    Kokkos::deep_copy(fixture.velocity_dot, dot);
    const auto minus_result = fixture.evaluate(EvaluationRequest::residual);
    const auto minus =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), minus_result.residual);
    for (int e = 0; e < 2; ++e) {
      for (int row = 0; row < Topology::dofs; ++row) {
        INFO("dimension = " << Dim << ", element = " << e << ", row = " << row
                            << ", column = " << column << ", transient = " << transient);
        check_close(jacobian(e, row, column), (plus(e, row) - minus(e, row)) / (2.0 * step), 1.e-7,
                    2.e-9);
      }
      u(e, column) += step;
      if (column < Topology::velocity_dofs) {
        dot(e, column) += step * fixture.time_coefficient;
      }
    }
  }
}

template <int Dim> void check_analytic_element(bool transient) {
  using Topology = Q2Q1<Dim>;
  ElementFixture<Dim> fixture(transient);
  auto x = Kokkos::create_mirror_view(fixture.x);
  auto u = Kokkos::create_mirror_view(fixture.solution);
  auto dot = Kokkos::create_mirror_view(fixture.velocity_dot);
  auto material = Kokkos::create_mirror_view(fixture.materials);
  auto flow = Kokkos::create_mirror_view(fixture.expressions);
  // Use the identity map and affine physical fields so all integrals are analytic.
  const auto velocity_gradient = [](int a, int d) { return 0.17 + 0.11 * a - 0.09 * d; };
  const auto dot_gradient = [](int a, int d) { return -0.08 + 0.04 * a + 0.07 * d; };
  for (int e = 0; e < 2; ++e) {
    material(e).density = 1.7;
    material(e).viscosity = 0.27;
    flow(e).transient = transient;
    flow(e).continuity = {true, 1.0};
    for (int a = 0; a < Dim; ++a) {
      material(e).source[a] = 0.6 - 0.2 * a;
      flow(e).momentum[a] = {{true, 1.0}, {true, 1.0}, {true, 1.0}, {true, 1.0}};
    }
    for (int node = 0; node < Topology::velocity_nodes; ++node) {
      for (int d = 0; d < Dim; ++d) {
        x(e, node, d) = Topology::node_coordinate(node, d);
      }
      for (int a = 0; a < Dim; ++a) {
        double value = 0.4 - 0.2 * a;
        double time_derivative = -0.2 + 0.1 * a;
        for (int d = 0; d < Dim; ++d) {
          value += velocity_gradient(a, d) * x(e, node, d);
          time_derivative += dot_gradient(a, d) * x(e, node, d);
        }
        u(e, a * Topology::velocity_nodes + node) = value;
        dot(e, a * Topology::velocity_nodes + node) = time_derivative;
      }
    }
    for (int node = 0; node < Topology::pressure_nodes; ++node) {
      u(e, Topology::velocity_dofs + node) = 1.8;
    }
  }
  Kokkos::deep_copy(fixture.x, x);
  Kokkos::deep_copy(fixture.solution, u);
  Kokkos::deep_copy(fixture.velocity_dot, dot);
  Kokkos::deep_copy(fixture.materials, material);
  Kokkos::deep_copy(fixture.expressions, flow);
  fixture.geometry = prepare_q2q1_geometry<Dim>(fixture.x);
  const auto output = fixture.evaluate(EvaluationRequest::residual);
  const auto status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output.status);
  const auto residual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output.residual);
  for (int e = 0; e < 2; ++e) {
    REQUIRE(status(e) == 0);
    for (int node = 0; node < Topology::velocity_nodes; ++node) {
      double integral = 1.0;
      double moment[Dim];
      double derivative[Dim];
      for (int d = 0; d < Dim; ++d) {
        integral *= Topology::node_coordinate(node, d) == 0 ? 4.0 / 3.0 : 1.0 / 3.0;
      }
      for (int d = 0; d < Dim; ++d) {
        derivative[d] = Topology::node_coordinate(node, d);
        for (int r = 0; r < Dim; ++r) {
          if (r != d) {
            derivative[d] *= Topology::node_coordinate(node, r) == 0 ? 4.0 / 3.0 : 1.0 / 3.0;
          }
        }
        moment[d] = derivative[d] / 3.0;
      }
      for (int a = 0; a < Dim; ++a) {
        double expected = material(e).source[a] * integral;
        if (transient) {
          double mass = (-0.2 + 0.1 * a) * integral;
          for (int d = 0; d < Dim; ++d) {
            mass += dot_gradient(a, d) * moment[d];
          }
          expected -= material(e).density * mass;
        }
        for (int d = 0; d < Dim; ++d) {
          double integrated_velocity = (0.4 - 0.2 * d) * integral;
          for (int r = 0; r < Dim; ++r) {
            integrated_velocity += velocity_gradient(d, r) * moment[r];
          }
          expected -= material(e).density * velocity_gradient(a, d) * integrated_velocity;
          const double stress =
              material(e).viscosity * (velocity_gradient(a, d) + velocity_gradient(d, a)) -
              (a == d ? 1.8 : 0.0);
          expected -= derivative[d] * stress;
        }
        check_close(residual(e, a * Topology::velocity_nodes + node), expected);
      }
    }
    double divergence = 0.0;
    for (int d = 0; d < Dim; ++d) {
      divergence += velocity_gradient(d, d);
    }
    for (int node = 0; node < Topology::pressure_nodes; ++node) {
      check_close(residual(e, Topology::velocity_dofs + node), divergence);
    }
  }
}

template <int Dim> void check_invalid_geometry() {
  using Topology = Q2Q1<Dim>;
  ElementFixture<Dim> fixture(true);
  for (int failure = 0; failure < 3; ++failure) {
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fixture.x);
    for (int node = 0; node < Topology::velocity_nodes; ++node) {
      for (int d = 0; d < Dim; ++d) {
        host(1, node, d) = Topology::node_coordinate(node, d);
      }
      host(1, node, 0) *= failure == 0 ? -1.0 : (failure == 1 ? 0.0 : 1.0);
    }
    if (failure == 2) {
      host(1, 0, 0) = std::numeric_limits<double>::quiet_NaN();
    }
    Kokkos::deep_copy(fixture.x, host);
    fixture.geometry = prepare_q2q1_geometry<Dim>(fixture.x);
    const auto output = fixture.evaluate(EvaluationRequest::both);
    const auto status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output.status);
    const auto residual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output.residual);
    const auto jacobian = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output.jacobian);
    CHECK(status(0) == 0);
    const auto expected =
        failure == 2 ? ElementStatus::nonfinite_geometry : ElementStatus::nonpositive_jacobian;
    CHECK(status(1) == static_cast<int>(expected));
    for (int row = 0; row < Topology::dofs; ++row) {
      CHECK(residual(1, row) == 0.0);
      for (int column = 0; column < Topology::dofs; ++column) {
        CHECK(jacobian(1, row, column) == 0.0);
      }
    }
  }
}

TEST_CASE("Q2/Q1 quad reference data matches legacy routines", "[assembly][kokkos][basis]") {
  check_reference_basis<2>();
}
TEST_CASE("Q2/Q1 hex reference data matches legacy routines", "[assembly][kokkos][basis]") {
  check_reference_basis<3>();
}
TEST_CASE("Q2 quadrilateral geometry preserves the physical chain rule",
          "[assembly][kokkos][geometry]") {
  check_geometry<2>();
}
TEST_CASE("Q2 hexahedral geometry preserves the physical chain rule",
          "[assembly][kokkos][geometry]") {
  check_geometry<3>();
}
TEST_CASE("Q2/Q1 quad element Jacobian matches finite differences", "[assembly][kokkos][element]") {
  check_element_derivatives<2>(GENERATE(false, true));
}
TEST_CASE("Q2/Q1 hex element Jacobian matches finite differences", "[assembly][kokkos][element]") {
  check_element_derivatives<3>(GENERATE(false, true));
}
TEST_CASE("Q2/Q1 quadrilateral integrals match an analytic affine flow",
          "[assembly][kokkos][element]") {
  check_analytic_element<2>(GENERATE(false, true));
}
TEST_CASE("Q2/Q1 hexahedral integrals match an analytic affine flow",
          "[assembly][kokkos][element]") {
  check_analytic_element<3>(GENERATE(false, true));
}
TEST_CASE("Invalid quadrilateral geometry blocks element evaluation",
          "[assembly][kokkos][geometry]") {
  check_invalid_geometry<2>();
}
TEST_CASE("Invalid hexahedral geometry blocks element evaluation", "[assembly][kokkos][geometry]") {
  check_invalid_geometry<3>();
}

TEST_CASE("Q2/Q1 element requests validate extents and support empty batches",
          "[assembly][kokkos][element]") {
  using Topology = Q2Q1<2>;
  const auto empty_x = coordinates<2>(0);
  const auto empty_geometry = prepare_q2q1_geometry<2>(empty_x);
  const ElementFields empty_solution("empty solution", 0, Topology::dofs);
  const ElementFields empty_dot("empty dot", 0, Topology::velocity_dofs);
  const Kokkos::View<ConstantFluid<2> *> empty_materials("empty materials", 0);
  const Kokkos::View<FlowExpressions<2> *> empty_expressions("empty expressions", 0);
  const auto output =
      evaluate_q2q1_elements(empty_geometry, empty_solution, empty_dot, empty_materials,
                             empty_expressions, 0.0, EvaluationRequest::both);
  Kokkos::fence();
  CHECK(output.residual.size() == 0);
  CHECK(output.jacobian.size() == 0);
  CHECK(output.status.size() == 0);
  const ElementCoordinates wrong_x("wrong coordinates", 1, Topology::velocity_nodes - 1, 2);
  REQUIRE_THROWS_AS(prepare_q2q1_geometry<2>(wrong_x), std::invalid_argument);
  ElementFixture<2> fixture(false);
  REQUIRE_THROWS_AS(fixture.evaluate(static_cast<EvaluationRequest>(0)), std::invalid_argument);
  fixture.time_coefficient = std::numeric_limits<double>::infinity();
  REQUIRE_THROWS_AS(fixture.evaluate(EvaluationRequest::both), std::invalid_argument);
  fixture.time_coefficient = 1.73;
  fixture.velocity_dot = empty_dot;
  REQUIRE_THROWS_AS(fixture.evaluate(EvaluationRequest::both), std::invalid_argument);
}

TEST_CASE("Q2/Q1 element evaluation reports invalid inputs and numerical overflow",
          "[assembly][kokkos][element]") {
  const int failure = GENERATE(0, 1, 2, 3, 4);
  ElementFixture<2> fixture(true);
  auto material = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fixture.materials);
  auto solution = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fixture.solution);
  auto dot = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fixture.velocity_dot);
  auto flow = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fixture.expressions);
  if (failure == 0) {
    material(1).density = 0.0;
  } else if (failure == 1) {
    material(1).viscosity = -1.0;
  } else if (failure == 2) {
    solution(1, 0) = std::numeric_limits<double>::quiet_NaN();
  } else if (failure == 3) {
    flow(1).momentum[0].advection.multiplier = std::numeric_limits<double>::infinity();
  } else {
    material(1).density = 1.e250;
    for (int column = 0; column < Q2Q1<2>::velocity_dofs; ++column) {
      dot(1, column) = 1.e250;
    }
  }
  Kokkos::deep_copy(fixture.materials, material);
  Kokkos::deep_copy(fixture.solution, solution);
  Kokkos::deep_copy(fixture.velocity_dot, dot);
  Kokkos::deep_copy(fixture.expressions, flow);
  const auto output = fixture.evaluate(EvaluationRequest::both);
  const auto status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output.status);
  CHECK(status(0) == 0);
  const auto expected =
      failure == 4 ? ElementStatus::nonfinite_contribution : ElementStatus::invalid_input;
  CHECK(status(1) == static_cast<int>(expected));
}

TEST_CASE("Steady element evaluation ignores unused time derivative fields",
          "[assembly][kokkos][element]") {
  ElementFixture<2> fixture(false);
  const auto baseline = fixture.evaluate(EvaluationRequest::both);
  const auto expected = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), baseline.residual);
  Kokkos::deep_copy(fixture.velocity_dot, std::numeric_limits<double>::quiet_NaN());
  const auto output = fixture.evaluate(EvaluationRequest::both);
  const auto status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output.status);
  const auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output.residual);
  for (int e = 0; e < 2; ++e) {
    CHECK(status(e) == 0);
    for (int row = 0; row < Q2Q1<2>::dofs; ++row) {
      check_close(actual(e, row), expected(e, row));
    }
  }
}
} // namespace
