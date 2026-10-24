#include <Kokkos_Core.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include "linalg/sparse_matrix_tpetra.h"
#include "mm_kokkos_constraints.hpp"
#include "mm_kokkos_tpetra.hpp"
extern "C" {
extern int ProcID;
extern int Num_Proc;
#include "rf_bc_const.h"
}

namespace {
using namespace goma::assembly;

void close(double actual, double expected) {
  INFO("rank = " << ProcID << ", actual = " << actual << ", expected = " << expected);
  CHECK(std::abs(actual - expected) <= 1.e-11 + 1.e-10 * std::abs(expected));
}

struct MatrixFixture {
  g_GomaSparseMatrix storage{};
  GomaSparseMatrix wrapper = &storage;
  std::vector<GO> owned;

  MatrixFixture(const std::vector<GO> &global,
                bool empty_ranks = false,
                bool diagonal_only = false) {
    for (std::size_t i = global.size(); i-- > 0;) {
      if (empty_ranks ? ProcID == 0 : int(i % Num_Proc) == ProcID) {
        owned.push_back(global[i]);
      }
    }
    GomaSparseMatrix_Tpetra_Create(&wrapper);
    std::vector<GO> rows;
    std::vector<GO> columns;
    for (GO row : owned) {
      for (GO column : global) {
        if (!diagonal_only || column == row) {
          rows.push_back(row);
          columns.push_back(column);
        }
      }
    }
    g_tpetra_create_graph(wrapper, owned.size(), owned.data(), owned.size(), owned.data(),
                          rows.size(), global.size(), rows.data(), columns.data());
  }
  ~MatrixFixture() {
    Kokkos::fence();
    g_tpetra_destroy(wrapper);
  }
  TpetraSparseMatrix &adapter() { return *static_cast<TpetraSparseMatrix *>(wrapper->data); }
  void zero(double value = 0.0) { g_tpetra_put_scalar(wrapper, value); }
};

struct Contributions {
  Kokkos::View<GO **> gids;
  ElementFields residual;
  Kokkos::View<double ***> jacobian;
  Kokkos::View<int *> status;

  explicit Contributions(bool empty_ranks = false) {
    const int elements = empty_ranks && ProcID != 0 ? 0 : 2;
    gids = Kokkos::View<GO **>("scatter DOFs", elements, 3);
    residual = ElementFields("scatter residual", elements, 3);
    jacobian = Kokkos::View<double ***>("scatter jacobian", elements, 3, 3);
    status = Kokkos::View<int *>("scatter status", elements);
    auto g = Kokkos::create_mirror_view(gids);
    auto r = Kokkos::create_mirror_view(residual);
    auto j = Kokkos::create_mirror_view(jacobian);
    constexpr GO global[4] = {17, 4, 28, 9};
    for (int e = 0; e < elements; ++e) {
      for (int row = 0; row < 3; ++row) {
        g(e, row) = global[e + row];
        r(e, row) = 0.5 * (e + 1) + 0.07 * (row + 1);
        for (int column = 0; column < 3; ++column) {
          j(e, row, column) = 100 * (e + 1) + 10 * (row + 1) + column + 1;
        }
      }
    }
    Kokkos::deep_copy(gids, g);
    Kokkos::deep_copy(residual, r);
    Kokkos::deep_copy(jacobian, j);
  }
};

double expected_residual(GO gid) {
  constexpr GO global[] = {17, 4, 28, 9};
  double result = 0.0;
  for (int e = 0; e < 2; ++e) {
    for (int row = 0; row < 3; ++row) {
      if (global[e + row] == gid) {
        result += 0.5 * (e + 1) + 0.07 * (row + 1);
      }
    }
  }
  return result;
}

double expected_matrix(GO row_gid, GO column_gid) {
  constexpr GO global[] = {17, 4, 28, 9};
  double result = 0.0;
  for (int e = 0; e < 2; ++e) {
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        if (global[e + row] == row_gid && global[e + column] == column_gid) {
          result += 100 * (e + 1) + 10 * (row + 1) + column + 1;
        }
      }
    }
  }
  return result;
}

void check_values(MatrixFixture &fixture,
                  const Kokkos::View<double *> &residual,
                  double matrix_factor,
                  double residual_factor,
                  double matrix_base = 0.0,
                  double residual_base = 0.0) {
  const auto &matrix = *fixture.adapter().matrix;
  const auto graph = matrix.getCrsGraph()->getLocalGraphHost();
  const auto values = matrix.getLocalValuesHost(Tpetra::Access::ReadOnly);
  const auto r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), residual);
  for (std::size_t row = 0; row < fixture.owned.size(); ++row) {
    const GO row_gid = matrix.getRowMap()->getGlobalElement(row);
    close(r(row), residual_base + residual_factor * expected_residual(row_gid));
    for (auto offset = graph.row_map(row); offset < graph.row_map(row + 1); ++offset) {
      const GO column_gid = matrix.getColMap()->getGlobalElement(graph.entries(offset));
      close(values(offset), matrix_base + matrix_factor * expected_matrix(row_gid, column_gid));
    }
  }
}

void check_scatter(bool empty_ranks) {
  MatrixFixture fixture({17, 4, 28, 9}, empty_ranks);
  Contributions local(empty_ranks);
  TpetraScatterPlan plan(fixture.wrapper, local.gids);
  Kokkos::View<double *> residual("owned residual", fixture.owned.size());
  fixture.zero(0.25);
  Kokkos::deep_copy(residual, 0.5);
  for (int repetition = 0; repetition < 2; ++repetition) {
    plan.add(local.residual, local.jacobian, local.status, residual, EvaluationRequest::both);
  }
  fixture.adapter().matrix->endAssembly();
  check_values(fixture, residual, 2.0, 2.0, 0.25, 0.5);

  // Reacquire values after closing, reading on host, reopening, and zeroing.
  fixture.adapter().matrix->beginAssembly();
  fixture.zero();
  Kokkos::deep_copy(residual, 0.0);
  plan.add({}, local.jacobian, local.status, {}, EvaluationRequest::jacobian);
  fixture.adapter().matrix->endAssembly();
  check_values(fixture, residual, 1.0, 0.0);
  // A residual-only line search must leave a closed matrix untouched.
  plan.add(local.residual, {}, local.status, residual, EvaluationRequest::residual);
  check_values(fixture, residual, 1.0, 1.0);
  fixture.adapter().matrix->beginModify();
  fixture.zero();
  Kokkos::deep_copy(residual, 0.0);
  plan.add(local.residual, local.jacobian, local.status, residual, EvaluationRequest::both);
  fixture.adapter().matrix->endModify();
  check_values(fixture, residual, 1.0, 1.0);
}

TEST_CASE("Tpetra device scatter sums shared elements into owned rows with ghost columns",
          "[assembly][kokkos][scatter][mpi]") {
  check_scatter(false);
}
TEST_CASE("Tpetra graph and scatter handle ranks with no rows or elements",
          "[assembly][kokkos][scatter][mpi]") {
  check_scatter(true);
}

TEST_CASE("Tpetra scatter rejects failed elements collectively before modifying the system",
          "[assembly][kokkos][scatter][mpi]") {
  MatrixFixture fixture({17, 4, 28, 9});
  Contributions local;
  TpetraScatterPlan plan(fixture.wrapper, local.gids);
  Kokkos::View<double *> residual("owned residual", fixture.owned.size());
  fixture.zero(7.0);
  Kokkos::deep_copy(residual, 9.0);
  auto status = Kokkos::create_mirror_view(local.status);
  status(0) = ProcID == 0 ? static_cast<int>(ElementStatus::invalid_input) : 0;
  status(1) = 0;
  Kokkos::deep_copy(local.status, status);
  REQUIRE_THROWS_AS(
      plan.add(local.residual, local.jacobian, local.status, residual, EvaluationRequest::both),
      std::invalid_argument);
  check_values(fixture, residual, 0.0, 0.0, 7.0, 9.0);
  Kokkos::deep_copy(local.status, 0);
  Kokkos::View<const double **> residual_input = local.residual;
  if (ProcID == 0) {
    residual_input = {};
  }
  REQUIRE_THROWS_AS(
      plan.add(residual_input, local.jacobian, local.status, residual, EvaluationRequest::both),
      std::invalid_argument);
  check_values(fixture, residual, 0.0, 0.0, 7.0, 9.0);
  fixture.adapter().matrix->endAssembly();
  REQUIRE_THROWS_AS(
      plan.add(local.residual, local.jacobian, local.status, residual, EvaluationRequest::both),
      std::invalid_argument);
}

TEST_CASE("Tpetra scatter rejects missing graph entries and duplicate element DOFs collectively",
          "[assembly][kokkos][scatter][mpi]") {
  {
    MatrixFixture fixture({17, 4, 28, 9}, false, true);
    Contributions local;
    REQUIRE_THROWS_AS(TpetraScatterPlan(fixture.wrapper, local.gids), std::invalid_argument);
  }
  {
    MatrixFixture fixture({17, 4, 28, 9});
    Contributions local;
    auto gids = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), local.gids);
    if (ProcID == 0) {
      gids(0, 1) = gids(0, 0);
    }
    Kokkos::deep_copy(local.gids, gids);
    REQUIRE_THROWS_AS(TpetraScatterPlan(fixture.wrapper, local.gids), std::invalid_argument);
  }
}

TEST_CASE("Tpetra scatter detects replacement of its cached matrix",
          "[assembly][kokkos][scatter][mpi]") {
  MatrixFixture fixture({17, 4, 28, 9});
  Contributions local;
  TpetraScatterPlan plan(fixture.wrapper, local.gids);
  fixture.adapter().matrix =
      Teuchos::rcp(new Tpetra::FECrsMatrix<double, LO, GO>(fixture.adapter().crs_graph));
  fixture.adapter().matrix->beginAssembly();
  Kokkos::View<double *> residual("owned residual", fixture.owned.size());
  REQUIRE_THROWS_AS(
      plan.add(local.residual, local.jacobian, local.status, residual, EvaluationRequest::both),
      std::invalid_argument);
  check_values(fixture, residual, 0.0, 0.0);
}

TEST_CASE("Constrained Q2 Q1 elements scatter per-incident-element penalties into Tpetra",
          "[assembly][kokkos][scatter][mpi]") {
  using Topology = Q2Q1<2>;
  std::vector<GO> global;
  for (int i = 0; i < 36; ++i) {
    global.push_back(i);
  }
  MatrixFixture fixture(global);
  ElementCoordinates x("quad coordinates", 2, Topology::velocity_nodes, 2);
  ElementFields solution("quad solution", 2, Topology::dofs);
  ElementFields dot("quad dot", 2, Topology::velocity_dofs);
  ElementConstraints constraints("quad constraints", 2, Topology::dofs);
  Kokkos::View<GO **> gids("quad DOFs", 2, Topology::dofs);
  Kokkos::View<ConstantFluid<2> *> materials("quad materials", 2);
  Kokkos::View<FlowExpressions<2> *> expressions("quad expressions", 2);
  auto hx = Kokkos::create_mirror_view(x);
  auto hu = Kokkos::create_mirror_view(solution);
  auto hd = Kokkos::create_mirror_view(dot);
  auto hc = Kokkos::create_mirror_view(constraints);
  auto hg = Kokkos::create_mirror_view(gids);
  auto hm = Kokkos::create_mirror_view(materials);
  auto hf = Kokkos::create_mirror_view(expressions);
  for (int e = 0; e < 2; ++e) {
    hm(e).density = 1.3;
    hm(e).viscosity = 0.7;
    hf(e).transient = true;
    hf(e).continuity = {true, 1.0};
    for (int a = 0; a < 2; ++a) {
      hf(e).momentum[a] = {{true, 1.0}, {true, 1.0}, {true, 1.0}, {true, 1.0}};
    }
    for (int node = 0; node < Topology::velocity_nodes; ++node) {
      const int grid_x = Topology::node_coordinate(node, 0) + 1 + 2 * e;
      const int grid_y = Topology::node_coordinate(node, 1) + 1;
      hx(e, node, 0) = 0.5 * grid_x;
      hx(e, node, 1) = 0.5 * grid_y;
      for (int a = 0; a < 2; ++a) {
        const int row = a * Topology::velocity_nodes + node;
        hg(e, row) = a * 15 + grid_y * 5 + grid_x;
        hu(e, row) = 0.1 + 0.2 * a + 0.03 * (grid_y * 5 + grid_x);
        hd(e, row) = 0.2 + 0.01 * (grid_y * 5 + grid_x);
        if (grid_x == 2 && a == 0) {
          hc(e, row) = {DirichletMode::residual, 0.3};
        }
      }
      if (node < Topology::pressure_nodes) {
        const int row = Topology::velocity_dofs + node;
        hg(e, row) = 30 + (grid_y / 2) * 3 + grid_x / 2;
        hu(e, row) = 1.0 + 0.02 * (hg(e, row) - 30);
        if (hg(e, row) == 31) {
          hc(e, row) = {DirichletMode::residual, 1.2};
        }
      }
    }
  }
  Kokkos::deep_copy(x, hx);
  Kokkos::deep_copy(solution, hu);
  Kokkos::deep_copy(dot, hd);
  Kokkos::deep_copy(constraints, hc);
  Kokkos::deep_copy(gids, hg);
  Kokkos::deep_copy(materials, hm);
  Kokkos::deep_copy(expressions, hf);
  const auto geometry = prepare_q2q1_geometry<2>(x);
  const auto local = evaluate_q2q1_elements(geometry, solution, dot, materials, expressions, 1.7,
                                            EvaluationRequest::both);
  // Preserve unconstrained results in separate host allocations for the reference sum.
  auto reference_r = Kokkos::create_mirror(Kokkos::HostSpace(), local.residual);
  auto reference_j = Kokkos::create_mirror(Kokkos::HostSpace(), local.jacobian);
  Kokkos::deep_copy(reference_r, local.residual);
  Kokkos::deep_copy(reference_j, local.jacobian);
  for (int e = 0; e < 2; ++e) {
    for (int row = 0; row < Topology::dofs; ++row) {
      if (hc(e, row).mode == DirichletMode::none) {
        continue;
      }
      reference_r(e, row) = DIRICHLET_PENALTY * (hu(e, row) - hc(e, row).value);
      for (int column = 0; column < Topology::dofs; ++column) {
        reference_j(e, row, column) = row == column ? DIRICHLET_PENALTY : 0.0;
      }
    }
  }
  apply_q2q1_dirichlet(local, solution, constraints, EvaluationRequest::both, DIRICHLET_PENALTY);
  TpetraScatterPlan plan(fixture.wrapper, gids);
  Kokkos::View<double *> residual("owned quad residual", fixture.owned.size());
  fixture.zero();
  plan.add(local.residual, local.jacobian, local.status, residual, EvaluationRequest::both);
  fixture.adapter().matrix->endAssembly();
  const auto &matrix = *fixture.adapter().matrix;
  const auto graph = matrix.getCrsGraph()->getLocalGraphHost();
  const auto values = matrix.getLocalValuesHost(Tpetra::Access::ReadOnly);
  const auto r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), residual);
  for (std::size_t owned = 0; owned < fixture.owned.size(); ++owned) {
    const GO row_gid = matrix.getRowMap()->getGlobalElement(owned);
    double expected_r = 0.0;
    std::vector<double> expected_j(36, 0.0);
    for (int e = 0; e < 2; ++e) {
      for (int row = 0; row < Topology::dofs; ++row) {
        if (hg(e, row) != row_gid) {
          continue;
        }
        expected_r += reference_r(e, row);
        for (int column = 0; column < Topology::dofs; ++column) {
          expected_j[hg(e, column)] += reference_j(e, row, column);
        }
      }
    }
    close(r(owned), expected_r);
    for (auto offset = graph.row_map(owned); offset < graph.row_map(owned + 1); ++offset) {
      const GO column_gid = matrix.getColMap()->getGlobalElement(graph.entries(offset));
      close(values(offset), expected_j[column_gid]);
    }
    if (row_gid == 2 || row_gid == 7 || row_gid == 12 || row_gid == 31) {
      close(expected_j[row_gid], 2.0 * DIRICHLET_PENALTY);
    }
  }
}
} // namespace
