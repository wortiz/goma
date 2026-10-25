/* Shared host-only legacy metadata fixtures for Kokkos adapter tests. */
#ifndef GOMA_TESTS_KOKKOS_LEGACY_FIXTURE_HPP
#define GOMA_TESTS_KOKKOS_LEGACY_FIXTURE_HPP

#include <array>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>

#include "mm_kokkos_legacy.hpp"
extern "C" {
#include "mm_as_const.h"
#include "rf_node_const.h"
#include "rf_solver_const.h"
}

namespace goma::assembly {

// One supported Q2/Q1 element, with reversed connectivity, and its initialized metadata.
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

// Process-global Nodes table for the fixture element. Local unknowns follow
// `order`, as Goma places owned nodes before external nodes. DBC offsets start
// unconstrained. Installs itself as Nodes and restores the previous table on exit.
template <int Dim> struct LegacyNodeTable {
  static constexpr int count = Q2Q1<Dim>::velocity_nodes;
  std::array<NODE_INFO_STRUCT, count> nodes{};
  std::array<NODE_INFO_STRUCT *, count> pointers{};
  std::array<NODAL_VARS_STRUCT, count> variables{};
  std::array<NODAL_VARS_STRUCT *, count> variable_pointers{};
  std::array<std::array<VARIABLE_DESCRIPTION_STRUCT, Dim + 1>, count> descriptors{};
  std::array<std::array<VARIABLE_DESCRIPTION_STRUCT *, Dim + 1>, count> descriptor_pointers{};
  std::array<std::array<int, Dim + 1>, count> offsets{};
  std::array<std::array<short, Dim + 1>, count> dbc{};
  std::array<short *, count> dbc_pointers{};
  std::array<int, count> first{};
  int unknowns = 0;
  NODE_INFO_STRUCT **saved = Nodes;

  LegacyNodeTable(const LegacyFixture<Dim> &fixture, std::span<const int> order) {
    if (order.size() != count) {
      throw std::invalid_argument("Node order must list every fixture node once");
    }
    for (int n : order) {
      auto &vars = variables[n];
      vars.Num_Unknowns = vars.Num_Var_Desc = Dim + (corner(fixture, n) ? 1 : 0);
      vars.Var_Desc_List = descriptor_pointers[n].data();
      vars.Nodal_Offset = offsets[n].data();
      first[n] = unknowns;
      unknowns += vars.Num_Unknowns;
      for (int v = 0; v < vars.Num_Unknowns; ++v) {
        auto &description = descriptors[n][v];
        description.Variable_Type = v < Dim ? VELOCITY1 + v : PRESSURE;
        description.Ndof = 1;
        description.MatID = -1;
        vars.Num_Var_Desc_Per_Type[description.Variable_Type] = 1;
        descriptor_pointers[n][v] = &description;
        offsets[n][v] = v;
        dbc[n][v] = -1;
      }
      variable_pointers[n] = &vars;
      dbc_pointers[n] = dbc[n].data();
      nodes[n].Nodal_Vars_Info = &variable_pointers[n];
      nodes[n].First_Unknown = &first[n];
      nodes[n].DBC = &dbc_pointers[n];
      pointers[n] = &nodes[n];
    }
    Nodes = pointers.data();
  }
  ~LegacyNodeTable() { Nodes = saved; }
  LegacyNodeTable(const LegacyNodeTable &) = delete;
  LegacyNodeTable &operator=(const LegacyNodeTable &) = delete;

  static bool corner(const LegacyFixture<Dim> &fixture, int node) {
    for (int c = 0; c < Q2Q1<Dim>::pressure_nodes; ++c) {
      if (fixture.connectivity[c] == node) {
        return true;
      }
    }
    return false;
  }
  int index(int node, int variable) const {
    return first[node] + (variable == PRESSURE ? Dim : variable - VELOCITY1);
  }
};

} // namespace goma::assembly
#endif
