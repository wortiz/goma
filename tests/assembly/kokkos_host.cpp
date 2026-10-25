#include <catch2/catch_test_macros.hpp>
#include <numeric>

#include "kokkos_legacy_fixture.hpp"

namespace {
using namespace goma::assembly;

bool contains(const std::vector<std::string> &issues, const std::string &text) {
  for (const auto &issue : issues) {
    if (issue.find(text) != std::string::npos) {
      return true;
    }
  }
  return false;
}

template <int Dim> void check_host_mapping() {
  using Topology = Q2Q1<Dim>;
  LegacyFixture<Dim> fixture;
  REQUIRE(check_legacy_flow(fixture.input()).empty());
  const std::size_t unknowns = 2 * Topology::dofs + 3;
  std::vector<GomaGlobalOrdinal> global(unknowns);
  std::vector<DirichletConstraint> constraints(unknowns);
  std::vector<double> solution(unknowns), dot(unknowns);
  for (std::size_t i = 0; i < unknowns; ++i) {
    global[i] = 1000 + 17 * i;
    solution[i] = 0.4 + 0.01 * i;
    dot[i] = -0.7 + 0.02 * i;
  }
  constraints[2 * Topology::dofs - 1] = {DirichletMode::hard_set, 0.3};
  const LegacyDofResolver resolve = [&](int node, int variable, int material) {
    CHECK(material == 0);
    return fixture.index(node, variable);
  };
  const auto snapshot = snapshot_legacy_flow<Dim>(fixture.input(), global, constraints, resolve);
  REQUIRE(snapshot.elements == 1);
  REQUIRE(snapshot.unknowns == unknowns);
  REQUIRE(snapshot.local_dofs.size() == Topology::dofs);
  CHECK(snapshot.material.density == fixture.material->density);
  CHECK(snapshot.material.viscosity == fixture.material->viscosity);
  CHECK(snapshot.expressions.transient);
  CHECK(snapshot.expressions.continuity.multiplier == 0.9);
  CHECK(snapshot.expressions.momentum[0].mass.enabled);
  CHECK(snapshot.expressions.momentum[0].stress.multiplier ==
        fixture.problem->etm[0][R_MOMENTUM1][LOG2_DIFFUSION]);
  CHECK(snapshot.constraints[0].mode == DirichletMode::hard_set);
  for (int row = 0; row < Topology::dofs; ++row) {
    CHECK(snapshot.local_dofs[row] == 2 * (Topology::dofs - row - 1) + 1);
    CHECK(snapshot.global_dofs[row] == global[snapshot.local_dofs[row]]);
  }
  for (int node = 0; node < Topology::velocity_nodes; ++node) {
    for (int d = 0; d < Dim; ++d) {
      CHECK(snapshot.coordinates[node * Dim + d] ==
            0.3 + 0.5 * d + Topology::node_coordinate(node, d));
    }
  }
  HostFieldMap<Dim> map(snapshot.local_dofs, snapshot.elements, snapshot.unknowns);
  const auto fields = map.gather(solution, dot, true);
  const auto u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fields.solution);
  const auto udot = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fields.velocity_dot);
  for (int row = 0; row < Topology::dofs; ++row) {
    CHECK(u(0, row) == solution[snapshot.local_dofs[row]]);
    if (row < Topology::velocity_dofs) {
      CHECK(udot(0, row) == dot[snapshot.local_dofs[row]]);
    }
  }
  const double saved = solution[snapshot.local_dofs[0]];
  solution[snapshot.local_dofs[0]] = -11.0;
  CHECK(u(0, 0) == saved); // No retained host span or pointer.
  const auto steady = map.gather(solution, {}, false);
  const auto steady_dot =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), steady.velocity_dot);
  CHECK(steady_dot(0, 0) == 0.0);
  REQUIRE_THROWS_AS(map.gather({}, dot, true), std::invalid_argument);
  REQUIRE_THROWS_AS(map.gather(solution, {}, true), std::invalid_argument);
  fixture.material->density = 9.0;
  CHECK(snapshot.material.density == 1.7);
}

TEST_CASE("Legacy Q2 Q1 host snapshots gather noncontiguous quad DOFs",
          "[assembly][kokkos][host]") {
  check_host_mapping<2>();
}
TEST_CASE("Legacy Q2 Q1 host snapshots gather noncontiguous hex DOFs", "[assembly][kokkos][host]") {
  check_host_mapping<3>();
}

TEST_CASE("Kokkos capability checks aggregate independent legacy incompatibilities",
          "[assembly][kokkos][preflight]") {
  LegacyFixture<2> fixture;
  fixture.update->AutoDiff = 1;
  fixture.update->Total_Num_Matrices = 2;
  fixture.problem->CoordinateSystem = CYLINDRICAL;
  fixture.problem->v[0][TEMPERATURE] = V_SOLNVECTOR;
  fixture.problem->e[0][R_ENERGY] = T_DIFFUSION;
  fixture.problem->i[0][PRESSURE] = I_Q2;
  fixture.material->DensityModel = USER;
  fixture.material->Mwt_funcModel = SUPG;
  fixture.options.pspg = true;
  fixture.options.matrix_format = "msr";
  fixture.options.linear_solver = AZTEC;
  fixture.boundaries[1].BC_Name = FLOW_PRESSURE_BC;
  fixture.types[0] = BILINEAR_QUAD;
  const auto issues = check_legacy_flow(fixture.input());
  CHECK(contains(issues, "AutoDiff"));
  CHECK(contains(issues, "single coupled matrix"));
  CHECK(contains(issues, "Cartesian"));
  CHECK(contains(issues, "unsupported variable"));
  CHECK(contains(issues, "unsupported equation"));
  CHECK(contains(issues, "pressure/test bases Q1"));
  CHECK(contains(issues, "density"));
  CHECK(contains(issues, "no SUPG"));
  CHECK(contains(issues, "PSPG"));
  CHECK(contains(issues, "tpetra"));
  CHECK(contains(issues, "stratimikos"));
  CHECK(contains(issues, "explicit nodal P"));
  CHECK(contains(issues, "9-node quads"));
}

TEST_CASE("Host snapshots reject malformed mappings before device access",
          "[assembly][kokkos][host]") {
  LegacyFixture<2> fixture;
  std::vector<GomaGlobalOrdinal> global(50);
  std::iota(global.begin(), global.end(), 0);
  std::vector<DirichletConstraint> constraints(50);
  REQUIRE_THROWS_AS(snapshot_legacy_flow<2>(fixture.input(), global, constraints,
                                            [](int, int, int) { return -1; }),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(snapshot_legacy_flow<2>(fixture.input(), global, constraints,
                                            [](int, int, int) { return 0; }),
                    std::invalid_argument);
  std::vector<int> indices(Q2Q1<2>::dofs, 0);
  indices[0] = 99;
  REQUIRE_THROWS_AS((HostFieldMap<2>(indices, 1, 50)), std::invalid_argument);
  REQUIRE_THROWS_AS((HostFieldMap<2>({}, 1, 50)), std::invalid_argument);
  HostFieldMap<2> empty({}, 0, 0);
  const auto fields = empty.gather({}, {}, true);
  CHECK(fields.solution.size() == 0);
  CHECK(fields.velocity_dot.size() == 0);
}

TEST_CASE("Legacy DBC offsets map hard-set and residual constraints", "[assembly][kokkos][host]") {
  LegacyFixture<2> fixture;
  fixture.boundaries[1].BC_relax = 0.5;
  fixture.boundaries[1].BC_Data_Float[0] = 2.1;
  NODE_INFO_STRUCT node{};
  NODAL_VARS_STRUCT vars{};
  vars.Num_Unknowns = 3;
  NODAL_VARS_STRUCT *metadata[] = {&vars};
  int first[] = {2};
  short offsets[] = {0, -1, 1};
  short *dbc[] = {offsets};
  node.Nodal_Vars_Info = metadata;
  node.First_Unknown = first;
  node.DBC = dbc;
  std::array<NODE_INFO_STRUCT *, 1> nodes{&node};
  const auto constraints = legacy_flow_constraints(nodes, fixture.boundaries, 8);
  CHECK(constraints[2].mode == DirichletMode::hard_set);
  CHECK(constraints[2].value == 0.3);
  CHECK(constraints[3].mode == DirichletMode::none);
  CHECK(constraints[4].mode == DirichletMode::residual);
  CHECK(constraints[4].value == 2.1);
  offsets[0] = 99;
  REQUIRE_THROWS_AS(legacy_flow_constraints(nodes, fixture.boundaries, 8), std::invalid_argument);
  first[0] = 7;
  REQUIRE_THROWS_AS(legacy_flow_constraints(nodes, fixture.boundaries, 8), std::invalid_argument);
}
TEST_CASE("Legacy host snapshots accept ranks with no nodes or elements",
          "[assembly][kokkos][host]") {
  LegacyFixture<2> fixture;
  fixture.mesh.num_nodes = 0;
  fixture.mesh.num_elems = 0;
  fixture.mesh.x_coord = nullptr;
  fixture.mesh.y_coord = nullptr;
  fixture.counts[0] = 0;
  fixture.block_connectivity[0] = nullptr;
  const auto snapshot = snapshot_legacy_flow<2>(fixture.input(), {}, {}, [](int, int, int) -> int {
    throw std::runtime_error("Empty ranks must not resolve a DOF");
  });
  CHECK(snapshot.elements == 0);
  CHECK(snapshot.unknowns == 0);
  CHECK(snapshot.coordinates.empty());
  CHECK(snapshot.local_dofs.empty());
}
template <int Dim> void check_initialized_snapshot() {
  LegacyFixture<Dim> fixture;
  std::vector<int> order(Q2Q1<Dim>::velocity_nodes);
  std::iota(order.begin(), order.end(), 0);
  LegacyNodeTable<Dim> table(fixture, order);
  for (int n = 0; n < table.count; ++n) {
    for (int v = 0; v < table.variables[n].Num_Unknowns; ++v) {
      table.dbc[n][v] = v == Dim ? 1 : (v == 0 ? 0 : -1);
    }
  }
  const int unknowns = table.unknowns;
  std::vector<GomaGlobalOrdinal> global(unknowns);
  for (int i = 0; i < unknowns; ++i) {
    global[i] = 2000 + 3 * (unknowns - i);
  }
  g_GomaSparseMatrix matrix{};
  matrix.n_cols = unknowns;
  matrix.n_rows = unknowns - 2; // Include ghost IDs in the copied map.
  matrix.global_ids = global.data();
  const auto snapshot = snapshot_initialized_legacy_flow<Dim>(fixture.input(), &matrix);
  const auto constraints = legacy_flow_constraints(table.pointers, fixture.boundaries, unknowns);
  const auto expected = snapshot_legacy_flow<Dim>(fixture.input(), global, constraints,
                                                  [&](int node, int variable, int material) {
                                                    CHECK(material == 0);
                                                    return table.index(node, variable);
                                                  });
  CHECK(snapshot.local_dofs == expected.local_dofs);
  CHECK(snapshot.global_dofs == expected.global_dofs);
  REQUIRE(snapshot.constraints.size() == expected.constraints.size());
  for (std::size_t i = 0; i < snapshot.constraints.size(); ++i) {
    CHECK(snapshot.constraints[i].mode == expected.constraints[i].mode);
    CHECK(snapshot.constraints[i].value == expected.constraints[i].value);
  }
  table.descriptors[0][0].Ndof = 2;
  CHECK_THROWS_AS(snapshot_initialized_legacy_flow<Dim>(fixture.input(), &matrix),
                  std::invalid_argument);
  table.descriptors[0][0].Ndof = 1;
  table.variables[0].Var_Desc_List = nullptr;
  CHECK_THROWS_AS(snapshot_initialized_legacy_flow<Dim>(fixture.input(), &matrix),
                  std::invalid_argument);
  table.variables[0].Var_Desc_List = table.descriptor_pointers[0].data();
  matrix.global_ids = nullptr;
  CHECK_THROWS_AS(snapshot_initialized_legacy_flow<Dim>(fixture.input(), &matrix),
                  std::invalid_argument);
}

TEST_CASE("Initialized legacy snapshots use Index_Solution and owned/ghost IDs",
          "[assembly][kokkos][host]") {
  check_initialized_snapshot<2>();
  check_initialized_snapshot<3>();
}
} // namespace
