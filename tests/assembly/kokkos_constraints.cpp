#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>

#include "mm_kokkos_constraints.hpp"
extern "C" {
#include "rf_bc_const.h"
}

namespace {
using namespace goma::assembly;

template <int Dim> void check_constraints(EvaluationRequest request, double penalty) {
  using Topology = Q2Q1<Dim>;
  const bool residual = request != EvaluationRequest::jacobian;
  const bool jacobian = request != EvaluationRequest::residual;
  ElementResult<Dim> result;
  result.status = Kokkos::View<int *>("constraint status", 2);
  if (residual) {
    result.residual = ElementFields("constraint residual", 2, Topology::dofs);
    Kokkos::deep_copy(result.residual, -4.0);
  }
  if (jacobian) {
    result.jacobian =
        Kokkos::View<double ***>("constraint jacobian", 2, Topology::dofs, Topology::dofs);
    Kokkos::deep_copy(result.jacobian, 7.0);
  }
  ElementFields solution("constraint solution", 2, Topology::dofs);
  Kokkos::deep_copy(solution, 3.0);
  ElementConstraints constraints("constant constraints", 2, Topology::dofs);
  auto host = Kokkos::create_mirror_view(constraints);
  const int velocity_row = Topology::velocity_nodes + 1;
  const int pressure_row = Topology::velocity_dofs;
  for (int e = 0; e < 2; ++e) {
    host(e, velocity_row) = {DirichletMode::hard_set, 3.0};
    host(e, pressure_row) = {DirichletMode::residual, 1.5};
  }
  Kokkos::deep_copy(constraints, host);
  apply_q2q1_dirichlet(result, solution, constraints, request, penalty);
  const auto status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.status);
  const auto r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.residual);
  const auto j = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.jacobian);
  for (int e = 0; e < 2; ++e) {
    REQUIRE(status(e) == 0);
    for (int row = 0; row < Topology::dofs; ++row) {
      const bool constrained = row == velocity_row || row == pressure_row;
      if (residual) {
        const double expected =
            row == velocity_row ? 0.0 : (row == pressure_row ? 1.5 * penalty : -4.0);
        CHECK(r(e, row) == expected);
      }
      if (jacobian) {
        for (int column = 0; column < Topology::dofs; ++column) {
          CHECK(j(e, row, column) == (constrained ? (column == row ? penalty : 0.0) : 7.0));
        }
      }
    }
  }
  const auto unchanged = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution);
  CHECK(unchanged(0, velocity_row) == 3.0);
  CHECK(unchanged(0, pressure_row) == 3.0);
  // Failed elements retain their entire previous result, even constrained rows.
  Kokkos::deep_copy(result.status, static_cast<int>(ElementStatus::nonpositive_jacobian));
  if (residual) {
    Kokkos::deep_copy(result.residual, -8.0);
  }
  if (jacobian) {
    Kokkos::deep_copy(result.jacobian, 11.0);
  }
  apply_q2q1_dirichlet(result, solution, constraints, request, penalty);
  Kokkos::fence();
  if (residual) {
    const auto failed = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.residual);
    CHECK(failed(0, pressure_row) == -8.0);
  }
  if (jacobian) {
    const auto failed = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.jacobian);
    CHECK(failed(0, pressure_row, pressure_row) == 11.0);
  }
}

TEST_CASE("Q2/Q1 Dirichlet rows preserve legacy penalties and columns", "[assembly][kokkos][bc]") {
  for (auto request :
       {EvaluationRequest::both, EvaluationRequest::residual, EvaluationRequest::jacobian}) {
    check_constraints<2>(request, DIRICHLET_PENALTY);
    check_constraints<3>(request, DIRICHLET_PENALTY);
    check_constraints<2>(request, 3.7);
  }
}

TEST_CASE("Residual-mode Dirichlet rows match their prescribed Jacobians",
          "[assembly][kokkos][bc]") {
  using Topology = Q2Q1<2>;
  ElementResult<2> result;
  result.status = Kokkos::View<int *>("status", 1);
  result.residual = ElementFields("residual", 1, Topology::dofs);
  result.jacobian = Kokkos::View<double ***>("jacobian", 1, Topology::dofs, Topology::dofs);
  ElementFields solution("solution", 1, Topology::dofs);
  ElementConstraints constraints("constraints", 1, Topology::dofs);
  auto host = Kokkos::create_mirror_view(constraints);
  host(0, 2) = {DirichletMode::residual, 1.2};
  Kokkos::deep_copy(constraints, host);
  constexpr double penalty = 2.3;
  constexpr double step = 1.e-5;
  Kokkos::deep_copy(solution, 0.7 + step);
  apply_q2q1_dirichlet(result, solution, constraints, EvaluationRequest::both, penalty);
  const auto plus = Kokkos::create_mirror(Kokkos::HostSpace(), result.residual);
  Kokkos::deep_copy(plus, result.residual);
  Kokkos::deep_copy(solution, 0.7 - step);
  apply_q2q1_dirichlet(result, solution, constraints, EvaluationRequest::both, penalty);
  const auto minus = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.residual);
  const auto matrix = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.jacobian);
  CHECK(std::abs((plus(0, 2) - minus(0, 2)) / (2.0 * step) - matrix(0, 2, 2)) < 1.e-9);
}

TEST_CASE("Invalid Dirichlet inputs invalidate the element before row replacement",
          "[assembly][kokkos][bc]") {
  using Topology = Q2Q1<2>;
  ElementResult<2> result;
  result.status = Kokkos::View<int *>("status", 1);
  result.residual = ElementFields("residual", 1, Topology::dofs);
  result.jacobian = Kokkos::View<double ***>("jacobian", 1, Topology::dofs, Topology::dofs);
  ElementFields solution("solution", 1, Topology::dofs);
  ElementConstraints constraints("constraints", 1, Topology::dofs);
  auto host = Kokkos::create_mirror_view(constraints);
  for (bool invalid_mode : {false, true}) {
    Kokkos::deep_copy(result.status, 0);
    Kokkos::deep_copy(result.residual, 5.0);
    host(0, 0) = {DirichletMode::residual, 0.0};
    host(0, 1) = invalid_mode ? DirichletConstraint{static_cast<DirichletMode>(77), 0.0}
                              : DirichletConstraint{DirichletMode::residual,
                                                    std::numeric_limits<double>::quiet_NaN()};
    Kokkos::deep_copy(constraints, host);
    apply_q2q1_dirichlet(result, solution, constraints, EvaluationRequest::both, 1.0);
    const auto status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.status);
    const auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.residual);
    CHECK(status(0) == static_cast<int>(ElementStatus::invalid_input));
    CHECK(values(0, 0) == 5.0);
  }
  REQUIRE_THROWS_AS(
      apply_q2q1_dirichlet(result, solution, constraints, EvaluationRequest::both, -1.0),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      apply_q2q1_dirichlet(result, solution, constraints, static_cast<EvaluationRequest>(0), 1.0),
      std::invalid_argument);
}
TEST_CASE("Dirichlet replacement reports overflow and accepts empty batches",
          "[assembly][kokkos][bc]") {
  using Topology = Q2Q1<2>;
  ElementResult<2> result;
  result.status = Kokkos::View<int *>("status", 1);
  result.residual = ElementFields("residual", 1, Topology::dofs);
  ElementFields solution("solution", 1, Topology::dofs);
  ElementConstraints constraints("constraints", 1, Topology::dofs);
  Kokkos::deep_copy(solution, std::numeric_limits<double>::max());
  auto host = Kokkos::create_mirror_view(constraints);
  host(0, 0) = {DirichletMode::residual, -std::numeric_limits<double>::max()};
  Kokkos::deep_copy(constraints, host);
  apply_q2q1_dirichlet(result, solution, constraints, EvaluationRequest::residual, 2.0);
  const auto status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.status);
  CHECK(status(0) == static_cast<int>(ElementStatus::nonfinite_contribution));

  ElementResult<2> empty;
  empty.status = Kokkos::View<int *>("empty status", 0);
  empty.residual = ElementFields("empty residual", 0, Topology::dofs);
  const ElementFields empty_solution("empty solution", 0, Topology::dofs);
  const ElementConstraints empty_constraints("empty constraints", 0, Topology::dofs);
  apply_q2q1_dirichlet(empty, empty_solution, empty_constraints, EvaluationRequest::residual, 1.0);
  Kokkos::fence();
  CHECK(empty.status.size() == 0);
}
} // namespace
