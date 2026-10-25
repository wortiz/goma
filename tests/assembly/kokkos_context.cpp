#include <Kokkos_Core.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "kokkos_legacy_fixture.hpp"
#include "linalg/sparse_matrix_tpetra.h"
#include "mm_kokkos_context.hpp"
#include "mm_kokkos_legacy.h"
extern "C" {
#include "rf_bc_const.h"
}
extern "C" {
extern int ProcID;
extern int Num_Proc;
}

namespace {
using namespace goma::assembly;

struct ContextHandle {
  goma_kokkos_context *value = nullptr;
  ContextHandle() { REQUIRE(goma_kokkos_context_create(&value) == GOMA_KOKKOS_SUCCESS); }
  ~ContextHandle() { CHECK(goma_kokkos_context_destroy(&value) == GOMA_KOKKOS_SUCCESS); }
};

template <int Dim> struct ContextFixture {
  using Topology = Q2Q1<Dim>;
  g_GomaSparseMatrix storage{};
  GomaSparseMatrix wrapper = &storage;
  std::vector<GO> owned;
  HostFlowSnapshot<Dim> snapshot;
  std::vector<double> solution;
  std::vector<double> dot;
  static int local_index(int row) { return Topology::dofs + 1 - row; }
  static GO gid(int row) { return 100 + 3 * row; }

  explicit ContextFixture(bool transient, bool empty_ranks = false) {
    for (int row = Topology::dofs - 1; row >= 0; --row) {
      if (empty_ranks ? ProcID == 0 : row % Num_Proc == ProcID) {
        owned.push_back(gid(row));
      }
    }
    GomaSparseMatrix_Tpetra_Create(&wrapper);
    std::vector<GO> rows, columns;
    for (GO row : owned) {
      for (int column = 0; column < Topology::dofs; ++column) {
        rows.push_back(row);
        columns.push_back(gid(column));
      }
    }
    g_tpetra_create_graph(wrapper, owned.size(), owned.data(), owned.size(), owned.data(),
                          rows.size(), Topology::dofs, rows.data(), columns.data());
    g_tpetra_put_scalar(wrapper, 0.0);
    snapshot.material.density = 1.4;
    snapshot.material.viscosity = 0.3;
    snapshot.expressions.transient = transient;
    snapshot.expressions.continuity = {true, 0.9};
    for (int a = 0; a < Dim; ++a) {
      snapshot.material.source[a] = 0.11 * (a + 1);
      snapshot.expressions.momentum[a] = {{true, 0.8}, {true, 0.7}, {true, 1.1}, {true, 0.6}};
    }
    if (empty_ranks && ProcID != 0) {
      return;
    }
    snapshot.elements = 1;
    snapshot.unknowns = Topology::dofs + 3;
    solution.resize(snapshot.unknowns);
    if (transient) {
      dot.resize(snapshot.unknowns, std::numeric_limits<double>::quiet_NaN());
    }
    for (int node = 0; node < Topology::velocity_nodes; ++node) {
      for (int d = 0; d < Dim; ++d) {
        snapshot.coordinates.push_back(Topology::node_coordinate(node, d));
      }
    }
    for (int row = 0; row < Topology::dofs; ++row) {
      snapshot.local_dofs.push_back(local_index(row));
      snapshot.global_dofs.push_back(gid(row));
      snapshot.constraints.push_back({});
      const bool velocity = row < Topology::velocity_dofs;
      if (velocity) {
        const int component = row / Topology::velocity_nodes;
        const int node = row % Topology::velocity_nodes;
        double value = 0.7 + 0.2 * component;
        for (int d = 0; d < Dim; ++d) {
          value += 0.15 * (component + 1) * (d + 1) * Topology::node_coordinate(node, d);
        }
        solution[local_index(row)] = value;
        if (transient) {
          dot[local_index(row)] = 0.1 + 0.03 * component + 0.01 * node;
        }
      } else {
        solution[local_index(row)] = 1.2 + 0.04 * (row - Topology::velocity_dofs);
      }
    }
    snapshot.constraints[0] = {DirichletMode::hard_set, 0.9};
    solution[local_index(0)] = 0.9;
    if (transient) {
      dot[local_index(0)] = 0.0;
    }
    snapshot.constraints[Topology::velocity_dofs] = {DirichletMode::residual, 1.0};
  }
  ~ContextFixture() {
    Kokkos::fence();
    g_tpetra_destroy(wrapper);
  }
  TpetraSparseMatrix &adapter() { return *static_cast<TpetraSparseMatrix *>(wrapper->data); }
};

template <int Dim> struct Reference {
  std::vector<double> residual = std::vector<double>(Q2Q1<Dim>::dofs, 0.0);
  std::vector<double> jacobian = std::vector<double>(Q2Q1<Dim>::dofs * Q2Q1<Dim>::dofs, 0.0);
};

template <int Dim>
Reference<Dim> reference(const HostFlowSnapshot<Dim> &snapshot,
                         const std::vector<double> &solution,
                         const std::vector<double> &dot,
                         double coefficient,
                         double penalty) {
  using Topology = Q2Q1<Dim>;
  Reference<Dim> expected;
  if (snapshot.elements == 0) {
    return expected;
  }
  ElementCoordinates coordinates("reference coordinates", 1, Topology::velocity_nodes, Dim);
  auto x = Kokkos::create_mirror_view(coordinates);
  for (int node = 0; node < Topology::velocity_nodes; ++node) {
    for (int d = 0; d < Dim; ++d) {
      x(0, node, d) = snapshot.coordinates[node * Dim + d];
    }
  }
  Kokkos::deep_copy(coordinates, x);
  ElementConstraints constraints("reference constraints", 1, Topology::dofs);
  auto c = Kokkos::create_mirror_view(constraints);
  for (int row = 0; row < Topology::dofs; ++row) {
    c(0, row) = snapshot.constraints[row];
  }
  Kokkos::deep_copy(constraints, c);
  Kokkos::View<ConstantFluid<Dim> *> materials("reference material", 1);
  auto m = Kokkos::create_mirror_view(materials);
  m(0) = snapshot.material;
  Kokkos::deep_copy(materials, m);
  Kokkos::View<FlowExpressions<Dim> *> expressions("reference expressions", 1);
  auto f = Kokkos::create_mirror_view(expressions);
  f(0) = snapshot.expressions;
  Kokkos::deep_copy(expressions, f);
  const auto geometry = prepare_q2q1_geometry<Dim>(coordinates);
  HostFieldMap<Dim> map(snapshot.local_dofs, 1, snapshot.unknowns);
  const auto fields = map.gather(solution, dot, snapshot.expressions.transient);
  const auto result =
      evaluate_q2q1_elements(geometry, fields.solution, fields.velocity_dot, materials, expressions,
                             coefficient, EvaluationRequest::both);
  apply_q2q1_dirichlet(result, fields.solution, constraints, EvaluationRequest::both, penalty);
  const auto r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.residual);
  const auto j = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.jacobian);
  for (int row = 0; row < Topology::dofs; ++row) {
    expected.residual[row] = r(0, row);
    for (int column = 0; column < Topology::dofs; ++column) {
      expected.jacobian[row * Topology::dofs + column] = j(0, row, column);
    }
  }
  return expected;
}

void close(double actual, double expected) {
  INFO("rank = " << ProcID << ", actual = " << actual << ", expected = " << expected);
  CHECK(std::abs(actual - expected) <= 1.e-12 + 1.e-10 * std::abs(expected));
}

template <int Dim>
void check_system(ContextFixture<Dim> &fixture,
                  const std::vector<double> &residual,
                  const Reference<Dim> &expected,
                  double matrix_factor,
                  double residual_factor,
                  double matrix_base = 0.0,
                  double residual_base = 0.0) {
  using Topology = Q2Q1<Dim>;
  const auto &matrix = *fixture.adapter().matrix;
  const auto graph = matrix.getCrsGraph()->getLocalGraphHost();
  const auto values = matrix.getLocalValuesHost(Tpetra::Access::ReadOnly);
  for (std::size_t row = 0; row < fixture.owned.size(); ++row) {
    const auto global_row = matrix.getRowMap()->getGlobalElement(row);
    const int local_row = (global_row - 100) / 3;
    close(residual[row], residual_base + residual_factor * expected.residual[local_row]);
    for (auto offset = graph.row_map(row); offset < graph.row_map(row + 1); ++offset) {
      const auto global_column = matrix.getColMap()->getGlobalElement(graph.entries(offset));
      const int local_column = (global_column - 100) / 3;
      close(values(offset),
            matrix_base +
                matrix_factor * expected.jacobian[local_row * Topology::dofs + local_column]);
    }
  }
}

template <int Dim> void check_context(bool transient, bool empty_ranks = false) {
  using Topology = Q2Q1<Dim>;
  ContextFixture<Dim> fixture(transient, empty_ranks);
  ContextHandle context;
  const auto snapshot = fixture.snapshot;
  constexpr double penalty = 3.7;
  setup_context(context.value, fixture.snapshot, fixture.wrapper, penalty);
  CHECK(goma_kokkos_runtime_check_finalize() == GOMA_KOKKOS_BAD_STATE);
  // Setup owns its metadata, rather than retaining spans or pointers to the source.
  fixture.snapshot = {};
  const auto expected = reference(snapshot, fixture.solution, fixture.dot, 1.73, penalty);
  std::vector<double> residual(fixture.owned.size(), 0.5);
  auto assemble = [&](goma_kokkos_evaluation_request request, double coefficient) {
    return goma_kokkos_context_assemble(
        context.value, fixture.solution.data(), fixture.solution.size(), fixture.dot.data(),
        fixture.dot.size(), coefficient, request, residual.data(), residual.size());
  };
  g_tpetra_put_scalar(fixture.wrapper, 0.25);
  CHECK(assemble(GOMA_KOKKOS_RESIDUAL_AND_JACOBIAN, 1.73) == GOMA_KOKKOS_SUCCESS);
  check_system(fixture, residual, expected, 1.0, 1.0, 0.25, 0.5);
  CHECK(std::strlen(goma_kokkos_context_last_error(context.value)) == 0);
  // Context evaluation neither changes FE phases nor zeroes existing contributions.
  CHECK(fixture.adapter().matrix->isFillActive());
  CHECK(assemble(GOMA_KOKKOS_RESIDUAL_AND_JACOBIAN, 1.73) == GOMA_KOKKOS_SUCCESS);
  check_system(fixture, residual, expected, 2.0, 2.0, 0.25, 0.5);
  fixture.adapter().matrix->endAssembly();
  CHECK(assemble(GOMA_KOKKOS_RESIDUAL, 1.73) == GOMA_KOKKOS_SUCCESS);
  CHECK(fixture.adapter().matrix->isFillComplete());
  check_system(fixture, residual, expected, 2.0, 3.0, 0.25, 0.5);
  fixture.adapter().matrix->beginAssembly();
  g_tpetra_put_scalar(fixture.wrapper, 0.0);
  CHECK(goma_kokkos_context_assemble(
            context.value, fixture.solution.data(), fixture.solution.size(), fixture.dot.data(),
            fixture.dot.size(), 1.73, GOMA_KOKKOS_JACOBIAN, nullptr, 0) == GOMA_KOKKOS_SUCCESS);
  check_system(fixture, residual, expected, 1.0, 3.0, 0.0, 0.5);
  // Refresh both field vectors and the Jacobian time coefficient without rebuilding geometry.
  for (int row = 1; row < Topology::dofs && snapshot.elements != 0; ++row) {
    fixture.solution[fixture.local_index(row)] += 0.03 * row;
    if (transient && row < Topology::velocity_dofs) {
      fixture.dot[fixture.local_index(row)] -= 0.01 * row;
    }
  }
  const auto updated = reference(snapshot, fixture.solution, fixture.dot, 2.31, penalty);
  std::fill(residual.begin(), residual.end(), 0.0);
  g_tpetra_put_scalar(fixture.wrapper, 0.0);
  CHECK(assemble(GOMA_KOKKOS_RESIDUAL_AND_JACOBIAN, 2.31) == GOMA_KOKKOS_SUCCESS);
  check_system(fixture, residual, updated, 1.0, 1.0);
  REQUIRE_THROWS_AS(setup_context(context.value, snapshot, fixture.wrapper, penalty),
                    std::invalid_argument);
  std::fill(residual.begin(), residual.end(), 0.0);
  g_tpetra_put_scalar(fixture.wrapper, 0.0);
  CHECK(assemble(GOMA_KOKKOS_RESIDUAL_AND_JACOBIAN, 2.31) == GOMA_KOKKOS_SUCCESS);
  check_system(fixture, residual, updated, 1.0, 1.0);
  CHECK(goma_kokkos_context_destroy(&context.value) == GOMA_KOKKOS_SUCCESS);
  CHECK(goma_kokkos_runtime_check_finalize() == GOMA_KOKKOS_SUCCESS);
  REQUIRE(goma_kokkos_context_create(&context.value) == GOMA_KOKKOS_SUCCESS);
  setup_context(context.value, snapshot, fixture.wrapper, penalty);
  std::fill(residual.begin(), residual.end(), 0.0);
  g_tpetra_put_scalar(fixture.wrapper, 0.0);
  CHECK(assemble(GOMA_KOKKOS_RESIDUAL_AND_JACOBIAN, 2.31) == GOMA_KOKKOS_SUCCESS);
  check_system(fixture, residual, updated, 1.0, 1.0);
}

TEST_CASE("Opaque quad contexts own setup data and refresh fields for each request",
          "[assembly][kokkos][context]") {
  check_context<2>(GENERATE(false, true));
}
TEST_CASE("Opaque hex contexts own setup data and refresh fields for each request",
          "[assembly][kokkos][context]") {
  check_context<3>(GENERATE(false, true));
}
TEST_CASE("Opaque contexts handle ranks without owned rows or elements",
          "[assembly][kokkos][context][mpi]") {
  check_context<2>(true, true);
}

TEST_CASE("Failed context setup is collective and can be retried",
          "[assembly][kokkos][context][mpi]") {
  ContextFixture<2> fixture(true);
  ContextHandle context;
  for (int error = 0; error < 3; ++error) {
    auto bad = fixture.snapshot;
    if (ProcID == 0) {
      if (error == 0) {
        bad.local_dofs[0] = -1;
      } else if (error == 1) {
        for (std::size_t i = 0; i < bad.coordinates.size(); i += 2) {
          bad.coordinates[i] *= -1.0;
        }
      } else {
        bad.global_dofs[0] = -11;
      }
    }
    REQUIRE_THROWS_AS(setup_context(context.value, bad, fixture.wrapper, 1.0),
                      std::invalid_argument);
    CHECK(std::strlen(goma_kokkos_context_last_error(context.value)) > 0);
    CHECK(goma_kokkos_context_assemble(context.value, nullptr, 0, nullptr, 0, 0.0,
                                       GOMA_KOKKOS_RESIDUAL, nullptr, 0) == GOMA_KOKKOS_BAD_STATE);
  }
  setup_context(context.value, fixture.snapshot, fixture.wrapper, 1.0);
  std::vector<double> residual(fixture.owned.size());
  CHECK(goma_kokkos_context_assemble(context.value, fixture.solution.data(),
                                     fixture.solution.size(), fixture.dot.data(),
                                     fixture.dot.size(), 1.7, GOMA_KOKKOS_RESIDUAL_AND_JACOBIAN,
                                     residual.data(), residual.size()) == GOMA_KOKKOS_SUCCESS);
}

TEST_CASE("Context errors are collective and leave global contributions unchanged",
          "[assembly][kokkos][context][mpi]") {
  ContextFixture<2> fixture(true);
  ContextHandle context;
  setup_context(context.value, fixture.snapshot, fixture.wrapper, 1.0);
  std::vector<double> residual(fixture.owned.size(), 9.0);
  const Reference<2> zero;
  g_tpetra_put_scalar(fixture.wrapper, 7.0);
  auto invoke = [&](std::size_t length, goma_kokkos_evaluation_request request) {
    return goma_kokkos_context_assemble(context.value, fixture.solution.data(), length,
                                        fixture.dot.data(), fixture.dot.size(), 1.7, request,
                                        residual.data(), residual.size());
  };
  CHECK(invoke(fixture.solution.size() - (ProcID == 0 ? 1 : 0),
               GOMA_KOKKOS_RESIDUAL_AND_JACOBIAN) == GOMA_KOKKOS_INVALID_ARGUMENT);
  check_system(fixture, residual, zero, 0.0, 0.0, 7.0, 9.0);
  const auto original = fixture.solution;
  if (ProcID == 0) {
    fixture.solution[fixture.local_index(0)] = -1.0; // Hard-set data was not imposed.
  }
  CHECK(invoke(fixture.solution.size(), GOMA_KOKKOS_RESIDUAL_AND_JACOBIAN) ==
        GOMA_KOKKOS_INVALID_ARGUMENT);
  check_system(fixture, residual, zero, 0.0, 0.0, 7.0, 9.0);
  fixture.solution = original;
  if (ProcID == 0) {
    fixture.solution[fixture.local_index(1)] = std::numeric_limits<double>::quiet_NaN();
  }
  CHECK(invoke(fixture.solution.size(), GOMA_KOKKOS_RESIDUAL_AND_JACOBIAN) ==
        GOMA_KOKKOS_INVALID_ARGUMENT);
  check_system(fixture, residual, zero, 0.0, 0.0, 7.0, 9.0);
  fixture.solution = original;
  if (Num_Proc > 1) {
    CHECK(invoke(fixture.solution.size(),
                 ProcID == 0 ? GOMA_KOKKOS_RESIDUAL : GOMA_KOKKOS_RESIDUAL_AND_JACOBIAN) ==
          GOMA_KOKKOS_INVALID_ARGUMENT);
    check_system(fixture, residual, zero, 0.0, 0.0, 7.0, 9.0);
  }
  fixture.adapter().matrix->endAssembly();
  CHECK(invoke(fixture.solution.size(), GOMA_KOKKOS_JACOBIAN) == GOMA_KOKKOS_INVALID_ARGUMENT);
  check_system(fixture, residual, zero, 0.0, 0.0, 7.0, 9.0);
  fixture.adapter().matrix->beginAssembly();
  CHECK(invoke(fixture.solution.size(), GOMA_KOKKOS_RESIDUAL_AND_JACOBIAN) == GOMA_KOKKOS_SUCCESS);
  CHECK(std::strlen(goma_kokkos_context_last_error(context.value)) == 0);
}

// One element held by every rank. Nodes are owned round-robin, owned nodes come
// first in the local vector, and global IDs are contiguous by owner, as in
// GomaSparseMatrix_SetProblemGraph. Non-owners hold the element as a ghost.
template <int Dim> struct LegacyContextFixture {
  using Topology = Q2Q1<Dim>;
  LegacyFixture<Dim> legacy;
  std::vector<int> order;
  std::unique_ptr<LegacyNodeTable<Dim>> table;
  std::vector<GO> global;
  std::size_t owned_rows = 0;
  g_GomaSparseMatrix storage{};
  GomaSparseMatrix wrapper = &storage;
  std::vector<double> x, xdot;
  int hard_node = -1, pressure_node = -1;

  static int owner(int node) { return node % Num_Proc; }

  explicit LegacyContextFixture(bool transient) {
    legacy.problem->TimeIntegration = transient ? TRANSIENT : STEADY;
    legacy.boundaries[1].BC_relax = 0.5; // Residual-mode pressure datum.
    legacy.boundaries[1].BC_Data_Float[0] = 2.1;
    constexpr int count = Topology::velocity_nodes;
    for (int pass = 0; pass < 2; ++pass) {
      for (int n = 0; n < count; ++n) {
        if ((owner(n) == ProcID) == (pass == 0)) {
          order.push_back(n);
        }
      }
    }
    table = std::make_unique<LegacyNodeTable<Dim>>(legacy, order);
    hard_node = legacy.connectivity[2];
    pressure_node = legacy.connectivity[1];
    table->dbc[hard_node][0] = 0;
    table->dbc[pressure_node][Dim] = 1;
    std::vector<GO> first(count);
    GO next = 0;
    for (int rank = 0; rank < Num_Proc; ++rank) {
      for (int n = 0; n < count; ++n) {
        if (owner(n) == rank) {
          first[n] = next;
          next += table->variables[n].Num_Unknowns;
        }
      }
    }
    global.resize(table->unknowns);
    for (int n = 0; n < count; ++n) {
      for (int v = 0; v < table->variables[n].Num_Unknowns; ++v) {
        global[table->first[n] + v] = first[n] + v;
      }
      if (owner(n) == ProcID) {
        owned_rows += table->variables[n].Num_Unknowns;
      }
    }
    GomaSparseMatrix_Tpetra_Create(&wrapper);
    std::vector<GO> rows, columns;
    for (std::size_t row = 0; row < owned_rows; ++row) {
      for (GO column : global) {
        rows.push_back(global[row]);
        columns.push_back(column);
      }
    }
    g_tpetra_create_graph(wrapper, owned_rows, global.data(), global.size(), global.data(),
                          rows.size(), global.size(), rows.data(), columns.data());
    storage.global_ids = global.data();
    storage.n_rows = owned_rows;
    storage.n_cols = global.size();
    // Ghost values equal the owner's values, as after exchange_dof.
    x.resize(global.size());
    xdot.resize(global.size());
    for (std::size_t i = 0; i < global.size(); ++i) {
      x[i] = 0.2 + 0.05 * global[i] + 0.002 * global[i] * global[i];
      xdot[i] = 0.1 - 0.01 * global[i];
    }
    x[table->index(hard_node, VELOCITY1)] = legacy.boundaries[0].BC_Data_Float[0];
    xdot[table->index(hard_node, VELOCITY1)] = 0.0;
  }
  ~LegacyContextFixture() {
    Kokkos::fence();
    g_tpetra_destroy(wrapper);
  }
  TpetraSparseMatrix &adapter() { return *static_cast<TpetraSparseMatrix *>(wrapper->data); }

  // Independent mapping through the explicit resolver rather than Index_Solution.
  HostFlowSnapshot<Dim> snapshot() const {
    const auto constraints =
        legacy_flow_constraints(table->pointers, legacy.boundaries, global.size());
    return snapshot_legacy_flow<Dim>(
        legacy.input(), global, constraints,
        [&](int node, int variable, int) { return table->index(node, variable); });
  }

  void check(const std::vector<double> &residual,
             const Reference<Dim> &expected,
             double matrix_base,
             double residual_base,
             double external_residual) {
    const auto element = snapshot();
    std::unordered_map<GO, int> row_of;
    for (int row = 0; row < Topology::dofs; ++row) {
      row_of[element.global_dofs[row]] = row;
    }
    const auto &matrix = *adapter().matrix;
    const auto graph = matrix.getCrsGraph()->getLocalGraphHost();
    const auto values = matrix.getLocalValuesHost(Tpetra::Access::ReadOnly);
    REQUIRE(matrix.getRowMap()->getLocalNumElements() == owned_rows);
    for (std::size_t row = 0; row < owned_rows; ++row) {
      REQUIRE(matrix.getRowMap()->getGlobalElement(row) == global[row]);
      const int element_row = row_of.at(global[row]);
      close(residual[row], residual_base + expected.residual[element_row]);
      for (auto offset = graph.row_map(row); offset < graph.row_map(row + 1); ++offset) {
        const auto column = matrix.getColMap()->getGlobalElement(graph.entries(offset));
        close(values(offset),
              matrix_base + expected.jacobian[element_row * Topology::dofs + row_of.at(column)]);
      }
    }
    for (std::size_t row = owned_rows; row < residual.size(); ++row) {
      CHECK(residual[row] == external_residual); // Ghost rows belong to their owners.
    }
  }
};

template <int Dim> void check_legacy_context(bool transient) {
  using Topology = Q2Q1<Dim>;
  LegacyContextFixture<Dim> fixture(transient);
  ContextHandle context;
  setup_legacy_context(context.value, fixture.legacy.input(), fixture.wrapper);
  CHECK(std::strlen(goma_kokkos_context_last_error(context.value)) == 0);
  const auto element = fixture.snapshot();
  REQUIRE(element.constraints[Topology::velocity_dofs + 1].mode == DirichletMode::residual);
  REQUIRE(element.constraints[2].mode == DirichletMode::hard_set);
  const double *dot = transient ? fixture.xdot.data() : nullptr;
  std::vector<double> residual(fixture.global.size(), 0.5);
  auto assemble = [&](goma_kokkos_evaluation_request request, double coefficient, double *r) {
    return goma_kokkos_context_assemble_legacy(context.value, fixture.wrapper, fixture.x.data(),
                                               dot, coefficient, request, r);
  };
  auto expected = reference(element, fixture.x, fixture.xdot, 1.9, DIRICHLET_PENALTY);
  g_tpetra_put_scalar(fixture.wrapper, 0.25);
  CHECK(assemble(GOMA_KOKKOS_RESIDUAL_AND_JACOBIAN, 1.9, residual.data()) == GOMA_KOKKOS_SUCCESS);
  fixture.check(residual, expected, 0.25, 0.5, 0.5);
  // Repeated updates refresh exchanged fields; Jacobian-only requests leave the residual.
  for (std::size_t i = 0; i < fixture.x.size(); ++i) {
    if (i != static_cast<std::size_t>(fixture.table->index(fixture.hard_node, VELOCITY1))) {
      fixture.x[i] -= 0.03 * fixture.global[i];
      fixture.xdot[i] += 0.02;
    }
  }
  expected = reference(element, fixture.x, fixture.xdot, 2.4, DIRICHLET_PENALTY);
  std::vector<double> before = residual;
  g_tpetra_put_scalar(fixture.wrapper, 0.0);
  CHECK(assemble(GOMA_KOKKOS_JACOBIAN, 2.4, nullptr) == GOMA_KOKKOS_SUCCESS);
  CHECK(residual == before);
  std::fill(residual.begin(), residual.begin() + fixture.owned_rows, 0.0);
  fixture.adapter().matrix->endAssembly();
  CHECK(assemble(GOMA_KOKKOS_RESIDUAL, 2.4, residual.data()) == GOMA_KOKKOS_SUCCESS);
  fixture.adapter().matrix->beginAssembly();
  fixture.check(residual, expected, 0.0, 0.0, 0.5);
  // Another wrapper is rejected on every rank before any contribution is added.
  g_GomaSparseMatrix other = fixture.storage;
  before = residual;
  CHECK(goma_kokkos_context_assemble_legacy(context.value, &other, fixture.x.data(), dot, 2.4,
                                            GOMA_KOKKOS_RESIDUAL,
                                            residual.data()) == GOMA_KOKKOS_INVALID_ARGUMENT);
  CHECK(residual == before);
  CHECK(std::strlen(goma_kokkos_context_last_error(context.value)) > 0);
}

TEST_CASE("Legacy-backed quad contexts assemble owned rows from Index_Solution metadata",
          "[assembly][kokkos][context][legacy][mpi]") {
  check_legacy_context<2>(GENERATE(false, true));
}
TEST_CASE("Legacy-backed hex contexts assemble owned rows from Index_Solution metadata",
          "[assembly][kokkos][context][legacy][mpi]") {
  check_legacy_context<3>(GENERATE(false, true));
}

TEST_CASE("Legacy context setup rejects one-rank failures collectively and is retryable",
          "[assembly][kokkos][context][legacy][mpi]") {
  LegacyContextFixture<2> fixture(true);
  ContextHandle context;
  auto rejected = [&](const std::string &text) {
    // INFO inside this lambda hangs nvcc 13.0 (cudafe++); CAPTURE reports the same text.
    CAPTURE(text);
    REQUIRE_THROWS_AS(setup_legacy_context(context.value, fixture.legacy.input(), fixture.wrapper),
                      std::invalid_argument);
    const std::string error = goma_kokkos_context_last_error(context.value);
    CHECK(
        (error.find(text) != std::string::npos || error.find("another rank") != std::string::npos));
    CHECK(goma_kokkos_context_assemble_legacy(context.value, fixture.wrapper, fixture.x.data(),
                                              fixture.xdot.data(), 1.0, GOMA_KOKKOS_RESIDUAL,
                                              nullptr) == GOMA_KOKKOS_INVALID_ARGUMENT);
  };
  // Pressure constraint absent from every owned DOF.
  constexpr int pressure_offset = 2;
  fixture.table->dbc[fixture.pressure_node][pressure_offset] = -1;
  rejected("pressure");
  fixture.table->dbc[fixture.pressure_node][pressure_offset] = 1;
  // Unsupported configuration on rank zero only.
  fixture.legacy.options.pspg = ProcID == 0;
  rejected("PSPG");
  fixture.legacy.options.pspg = false;
  // Host vector map disagrees with the Tpetra owned row map on rank zero.
  if (ProcID == 0 && fixture.owned_rows >= 2) {
    std::swap(fixture.global[0], fixture.global[1]);
  }
  rejected("row map");
  if (ProcID == 0 && fixture.owned_rows >= 2) {
    std::swap(fixture.global[0], fixture.global[1]);
  }
  // A local element included twice on rank zero.
  std::vector<int> repeated(fixture.legacy.connectivity.begin(), fixture.legacy.connectivity.end());
  repeated.insert(repeated.end(), repeated.begin(), repeated.end());
  if (ProcID == 0) {
    fixture.legacy.block_connectivity[0] = repeated.data();
    fixture.legacy.counts[0] = 2;
    fixture.legacy.mesh.num_elems = 2;
  }
  rejected("more than once");
  fixture.legacy.block_connectivity[0] = fixture.legacy.connectivity.data();
  fixture.legacy.counts[0] = 1;
  fixture.legacy.mesh.num_elems = 1;
  // The C entry point joins missing-mesh failures before collective setup.
  CHECK(goma_kokkos_context_setup_legacy(context.value, nullptr, fixture.wrapper) ==
        GOMA_KOKKOS_INVALID_ARGUMENT);
  CHECK(std::strlen(goma_kokkos_context_last_error(context.value)) > 0);
  setup_legacy_context(context.value, fixture.legacy.input(), fixture.wrapper);
  std::vector<double> residual(fixture.global.size(), 0.0);
  CHECK(goma_kokkos_context_assemble_legacy(
            context.value, fixture.wrapper, fixture.x.data(), fixture.xdot.data(), 1.0,
            GOMA_KOKKOS_RESIDUAL_AND_JACOBIAN, residual.data()) == GOMA_KOKKOS_SUCCESS);
  // Successful setup is immutable.
  REQUIRE_THROWS_AS(setup_legacy_context(context.value, fixture.legacy.input(), fixture.wrapper),
                    std::invalid_argument);
}
} // namespace
