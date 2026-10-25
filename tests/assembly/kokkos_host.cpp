#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <memory>
#include <numeric>

#include "mm_kokkos_legacy.hpp"
extern "C" {
#include "mm_as_const.h"
#include "rf_solver_const.h"
}

namespace {
using namespace goma::assembly;

template <int Dim> struct LegacyFixture {
  using Topology = Q2Q1<Dim>;
  std::unique_ptr<UPD_STRUCT> update = std::make_unique<UPD_STRUCT>();
  std::unique_ptr<PROBLEM_DESCRIPTION_STRUCT> problem =
      std::make_unique<PROBLEM_DESCRIPTION_STRUCT>();
  std::unique_ptr<MATRL_PROP_STRUCT> material = std::make_unique<MATRL_PROP_STRUCT>();
  std::unique_ptr<GEN_NEWT_STRUCT> constitutive = std::make_unique<GEN_NEWT_STRUCT>();
  std::array<PROBLEM_DESCRIPTION_STRUCT *, 1> problems{problem.get()};
  std::array<MATRL_PROP_STRUCT *, 1> materials{material.get()};
  std::array<GEN_NEWT_STRUCT *, 1> models{constitutive.get()};
  std::array<Boundary_Condition, 2> boundaries{};
  std::array<int, 1> block_materials{0};
  std::array<int, 1> ids{17}, types{Dim == 2 ? BIQUAD_QUAD : TRIQUAD_HEX};
  std::array<int, 1> counts{1}, nodes_per_element{Topology::velocity_nodes};
  std::array<int, Topology::velocity_nodes> connectivity{};
  std::array<int *, 1> block_connectivity{connectivity.data()};
  std::array<std::array<double, Topology::velocity_nodes>, Dim> coordinates{};
  Exo_DB mesh{};
  LegacyFlowOptions options;

  LegacyFixture() {
    update->Total_Num_Matrices = 1;
    update->Num_Mat = 1;
    problem->Num_Dim = Dim;
    problem->CoordinateSystem = CARTESIAN;
    problem->MeshMotion = ARBITRARY;
    problem->TimeIntegration = TRANSIENT;
    problem->IntegrationMap = I_Q2;
    material->DensityModel = CONSTANT;
    material->density = 1.7;
    material->ViscosityModel = CONSTANT;
    material->viscosity = 0.4;
    material->DilationalViscosityModel = DILVISCM_KAPPAWIPESMU;
    material->MomentumSourceModel = CONSTANT;
    material->Mwt_funcModel = GALERKIN;
    constitutive->ConstitutiveEquation = NEWTONIAN;
    for (int a = 0; a < Dim; ++a) {
      const int equation = R_MOMENTUM1 + a;
      problem->v[0][equation] = V_SOLNVECTOR;
      problem->e[0][equation] =
          T_SOMETHING | T_MASS | T_ADVECTION | T_BOUNDARY | T_DIFFUSION | T_SOURCE;
      problem->i[0][equation] = I_Q2;
      problem->w[0][equation] = I_Q2;
      for (int term = 0; term < MAX_TERM_TYPES; ++term) {
        problem->etm[0][equation][term] = 0.7 + 0.03 * term + 0.1 * a;
      }
      material->momentum_source[a] = 0.2 - 0.1 * a;
    }
    problem->v[0][PRESSURE] = V_SOLNVECTOR;
    problem->e[0][R_PRESSURE] = T_SOMETHING | T_ADVECTION;
    problem->i[0][PRESSURE] = I_Q1;
    problem->w[0][R_PRESSURE] = I_Q1;
    problem->etm[0][R_PRESSURE][LOG2_ADVECTION] = 0.9;
    boundaries[0].BC_Name = U_BC;
    boundaries[1].BC_Name = P_BC;
    for (auto &boundary : boundaries) {
      std::strcpy(boundary.Set_Type, "NS");
      boundary.BC_relax = -1.0;
      boundary.BC_Data_Float[0] = 0.3;
    }
    options.matrix_format = "tpetra";
    options.linear_solver = STRATIMIKOS;
    mesh.num_dim = Dim;
    mesh.num_elems = 1;
    mesh.num_nodes = Topology::velocity_nodes;
    mesh.num_elem_blocks = 1;
    mesh.eb_id = ids.data();
    mesh.eb_elem_itype = types.data();
    mesh.eb_num_nodes_per_elem = nodes_per_element.data();
    mesh.eb_num_elems = counts.data();
    mesh.eb_conn = block_connectivity.data();
    mesh.x_coord = coordinates[0].data();
    mesh.y_coord = coordinates[1].data();
    if constexpr (Dim == 3) {
      mesh.z_coord = coordinates[2].data();
    }
    for (int node = 0; node < Topology::velocity_nodes; ++node) {
      connectivity[node] = Topology::velocity_nodes - node - 1;
      for (int d = 0; d < Dim; ++d) {
        coordinates[d][connectivity[node]] = 0.3 + 0.5 * d + Topology::node_coordinate(node, d);
      }
    }
  }

  LegacyFlowInput input() const {
    return {mesh, *update, problems, materials, models, boundaries, block_materials, options};
  }

  int index(int node, int variable) const {
    int position = -1;
    for (int i = 0; i < Topology::velocity_nodes; ++i) {
      if (connectivity[i] == node) {
        position = i;
      }
    }
    if (position < 0 || (variable == PRESSURE && position >= Topology::pressure_nodes)) {
      throw std::invalid_argument("Unexpected host node/variable lookup");
    }
    const int row = variable == PRESSURE
                        ? Topology::velocity_dofs + position
                        : (variable - VELOCITY1) * Topology::velocity_nodes + position;
    return 2 * (Topology::dofs - 1 - row) + 1;
  }
};

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
} // namespace
