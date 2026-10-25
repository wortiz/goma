#include "mm_kokkos_legacy.hpp"
#include "mm_kokkos_context.hpp"
#include "mm_kokkos_context_internal.hpp"
#include "mm_kokkos_legacy.h"

#include "linalg/sparse_matrix_tpetra.h"

extern "C" {
#include "ac_particles.h"
#include "ac_stability.h"
#include "mm_as.h"
#include "mm_as_const.h"
#include "mm_mp.h"
#include "mm_unknown_map.h"
#include "rf_bc.h"
#include "rf_bc_const.h"
#include "rf_fem.h"
#include "rf_io.h"
#include "rf_solver.h"
#include "rf_solver_const.h"
}
#include <algorithm>
#include <cstdio>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

namespace goma::assembly {
namespace {
std::string legacy_name(int index, const EQUATION_NAMES_STRUCT *names, int count) {
  for (int i = 0; i < count; ++i) {
    if (names[i].Index == index && names[i].name1 != nullptr) {
      return std::string(names[i].name1) + " (" + std::to_string(index) + ")";
    }
  }
  return std::to_string(index);
}
bool velocity_variable(int variable, int dimension) {
  return variable >= VELOCITY1 && variable < VELOCITY1 + dimension;
}
bool constant_boundary(const Boundary_Condition &bc, int dimension) {
  return (bc.BC_Name == U_BC || bc.BC_Name == V_BC || bc.BC_Name == P_BC ||
          (dimension == 3 && bc.BC_Name == W_BC)) &&
         bc.Set_Type[0] == 'N' && bc.Set_Type[1] == 'S' && bc.Set_Type[2] == '\0' &&
         std::isfinite(bc.BC_Data_Float[0]) && std::isfinite(bc.BC_relax);
}
} // namespace

LegacyFlowInput describe_legacy_flow(const Exo_DB &exo) {
  if (upd == nullptr) {
    throw std::invalid_argument("Missing initialized legacy update metadata");
  }
  LegacyFlowInput input{exo, *upd, {}, {}, {}, {}, {}, {}};
  const auto materials = static_cast<std::size_t>(std::max(upd->Num_Mat, 0));
  if (pd_glob != nullptr && mp_glob != nullptr && gn_glob != nullptr) {
    input.problems = {pd_glob, materials};
    input.materials = {mp_glob, materials};
    input.constitutive = {gn_glob, materials};
  }
  if (BC_Types != nullptr && Num_BC > 0) {
    input.boundaries = {BC_Types, static_cast<std::size_t>(Num_BC)};
  }
  if (Matilda != nullptr && exo.num_elem_blocks > 0) {
    input.block_materials = {Matilda, static_cast<std::size_t>(exo.num_elem_blocks)};
  }
  auto &options = input.options;
  options.matrix_format = Matrix_Format;
  options.linear_solver = Linear_Solver;
  options.pspg = PSPG != 0;
  options.continuity_gls = Cont_GLS != 0;
  options.level_set = Use_Level_Set != 0 || Use_Phase_Field != 0;
  options.augmenting = nAC != 0;
  options.continuation = Continuation != ALC_NONE;
  options.linear_stability = Linear_Stability != LSA_NONE;
  options.particles = Particle_Dynamics != 0;
  options.rotations = Num_ROT != 0;
  options.element_pressure_datum = PRESSURE_DATUM != 0;
  options.numerical_jacobian = Debug_Flag < 0;
  return input;
}

std::vector<std::string> check_legacy_flow(const LegacyFlowInput &input) {
  std::vector<std::string> issues;
  const auto &update = input.update;
  const auto &options = input.options;
  auto require = [&](bool condition, const std::string &message) {
    if (!condition) {
      issues.push_back(message);
    }
  };
  require(input.mesh.num_elem_blocks >= 0 && input.mesh.num_elems >= 0 && input.mesh.num_nodes >= 0,
          "geometry: invalid mesh metadata counts");
  const int dimension = input.mesh.num_dim;
  require(dimension == 2 || dimension == 3, "geometry: only two or three dimensions are supported");
  require(update.Total_Num_Matrices == 1 && !update.SegregatedSolve,
          "solver: only a single coupled matrix is supported");
  require(update.Num_Mat == 1, "material: only one homogeneous fluid material is supported");
  require(!update.AutoDiff, "Jacobian: Use AutoDiff Assembly is not yet supported by Kokkos");
  require(options.matrix_format == "tpetra", "solver: Matrix Storage Format must be tpetra");
  require(options.linear_solver == STRATIMIKOS, "solver: Solution Algorithm must be stratimikos");
  require(!options.pspg && !update.PSPG_advection_correction, "stabilization: PSPG is unsupported");
  require(!options.continuity_gls, "stabilization: Continuity GLS is unsupported");
  require(!options.level_set, "physics: level-set and phase-field interfaces are unsupported");
  require(!options.xfem && !update.XFEM, "interpolation: XFEM and enriched bases are unsupported");
  require(!options.augmenting, "solver: augmenting conditions are unsupported");
  require(!options.continuation, "solver: continuation and hunting are unsupported");
  require(!options.linear_stability, "solver: linear-stability assembly is unsupported");
  require(!options.numerical_jacobian,
          "Jacobian: numerical Jacobian/debug assembly is unsupported");
  require(!options.particles, "physics: particle dynamics are unsupported");
  require(!options.rotations, "boundary: rotated equations are unsupported");
  require(!options.element_pressure_datum,
          "boundary: PRESSURE DATUM element constraints are unsupported; use a nodal P condition");
  require(input.problems.size() == static_cast<std::size_t>(std::max(update.Num_Mat, 0)) &&
              input.materials.size() == input.problems.size() &&
              input.constitutive.size() == input.problems.size(),
          "material: incomplete problem or property metadata");
  for (std::size_t m = 0; m < input.problems.size(); ++m) {
    const std::string prefix = "material " + std::to_string(m + 1) + ": ";
    if (m >= input.materials.size() || m >= input.constitutive.size() ||
        input.problems[m] == nullptr || input.materials[m] == nullptr ||
        input.constitutive[m] == nullptr) {
      issues.push_back(prefix + "missing initialized metadata");
      continue;
    }
    const auto &pd = *input.problems[m];
    const auto &mp = *input.materials[m];
    const int dim = pd.Num_Dim;
    require(dim == dimension, prefix + "geometry dimension differs from the mesh");
    require(pd.CoordinateSystem == CARTESIAN, prefix + "coordinates must be Cartesian");
    require(pd.MeshMotion == ARBITRARY, prefix + "only fixed Eulerian meshes are supported");
    require(pd.IntegrationMap == I_Q2, prefix + "geometry mapping must resolve to Q2");
    require(pd.TimeIntegration == STEADY || pd.TimeIntegration == TRANSIENT,
            prefix + "unsupported time integration scheme");
    require(mp.DensityModel == CONSTANT && std::isfinite(mp.density) && mp.density > 0.0,
            prefix + "density must be a positive finite constant");
    require(mp.ViscosityModel == CONSTANT && std::isfinite(mp.viscosity) && mp.viscosity >= 0.0 &&
                input.constitutive[m]->ConstitutiveEquation == NEWTONIAN,
            prefix + "viscosity must be finite, constant, and Newtonian");
    require(mp.DilationalViscosityModel == DILVISCM_KAPPAWIPESMU,
            prefix + "dilational-viscosity stress corrections are unsupported");
    require(mp.Mwt_funcModel == GALERKIN, prefix + "momentum weighting must be Galerkin (no SUPG)");
    require(mp.MomentumSourceModel == CONSTANT, prefix + "momentum source must be constant");
    for (int a = 0; a < std::min(std::max(dim, 0), 3); ++a) {
      require(std::isfinite(mp.momentum_source[a]), prefix + "nonfinite constant momentum source");
    }
    const int matrices = std::min(std::max(update.Total_Num_Matrices, 0), MAX_NUM_MATRICES);
    for (int matrix = 0; matrix < matrices; ++matrix) {
      for (int variable = 0; variable < MAX_VARIABLE_TYPES; ++variable) {
        const bool allowed = variable == PRESSURE || velocity_variable(variable, dim);
        if (pd.v[matrix][variable] != 0 && (!allowed || pd.v[matrix][variable] != V_SOLNVECTOR)) {
          issues.push_back(prefix + "unsupported variable/activity " +
                           legacy_name(variable, Var_Name, Num_Var_Names));
        }
      }
      for (int equation = 0; equation < MAX_EQNS; ++equation) {
        const int flags = pd.e[matrix][equation];
        if (flags == 0) {
          continue;
        }
        const bool velocity = velocity_variable(equation, dim);
        if (!velocity && equation != R_PRESSURE) {
          issues.push_back(prefix + "unsupported equation " +
                           legacy_name(equation, EQ_Name, Num_EQ_Names));
          continue;
        }
        const int allowed = T_SOMETHING | T_ADVECTION | T_BOUNDARY |
                            (velocity ? (T_MASS | T_DIFFUSION | T_SOURCE) : 0);
        require((flags & ~allowed) == 0, prefix + "unsupported equation terms for " +
                                             legacy_name(equation, EQ_Name, Num_EQ_Names));
        const int interpolation = velocity ? I_Q2 : I_Q1;
        require(pd.i[matrix][equation] == interpolation && pd.w[matrix][equation] == interpolation,
                prefix + "velocity/test bases must be Q2 and pressure/test bases Q1 (equation " +
                    legacy_name(equation, EQ_Name, Num_EQ_Names) + ")");
        for (int term = 0; term < MAX_TERM_TYPES; ++term) {
          if (flags & (1 << term)) {
            require(std::isfinite(pd.etm[matrix][equation][term]),
                    prefix + "nonfinite equation term multiplier");
          }
        }
      }
      for (int variable = 0; variable < MAX_VARIABLE_TYPES; ++variable) {
        if (variable == PRESSURE || velocity_variable(variable, dim)) {
          require(pd.e[matrix][variable] != 0 && pd.v[matrix][variable] == V_SOLNVECTOR,
                  prefix + "all velocity components and pressure must be solved together");
        }
      }
    }
  }
  require(input.block_materials.size() ==
              static_cast<std::size_t>(std::max(input.mesh.num_elem_blocks, 0)),
          "geometry: missing block-to-material map");
  for (int block = 0; block < input.mesh.num_elem_blocks; ++block) {
    const std::string prefix =
        "block " + std::to_string(input.mesh.eb_id ? input.mesh.eb_id[block] : block + 1) + ": ";
    require(static_cast<std::size_t>(block) < input.block_materials.size() &&
                input.block_materials[block] == 0,
            prefix + "every block must use the same active fluid material");
    require(input.mesh.eb_elem_itype && input.mesh.eb_num_nodes_per_elem &&
                ((dimension == 2 && input.mesh.eb_elem_itype[block] == BIQUAD_QUAD &&
                  input.mesh.eb_num_nodes_per_elem[block] == 9) ||
                 (dimension == 3 && input.mesh.eb_elem_itype[block] == TRIQUAD_HEX &&
                  input.mesh.eb_num_nodes_per_elem[block] == 27)),
            prefix +
                "only 9-node quads or 27-node hexes matching the spatial dimension are supported");
  }
  bool pressure = false;
  for (std::size_t bc = 0; bc < input.boundaries.size(); ++bc) {
    const auto &boundary = input.boundaries[bc];
    const bool supported = constant_boundary(boundary, dimension);
    require(supported, "boundary " + std::to_string(bc + 1) +
                           ": only finite constant nodal velocity/P conditions are supported");
    pressure = pressure || (supported && boundary.BC_Name == P_BC);
  }
  require(pressure, "boundary: an explicit nodal P constraint is required");
  std::sort(issues.begin(), issues.end());
  issues.erase(std::unique(issues.begin(), issues.end()), issues.end());
  return issues;
}

std::vector<DirichletConstraint>
legacy_flow_constraints(std::span<NODE_INFO_STRUCT *const> nodes,
                        std::span<const Boundary_Condition> boundaries,
                        std::size_t unknowns) {
  if (unknowns > INT_MAX) {
    throw std::invalid_argument("Legacy solution vector exceeds the local-index range");
  }
  std::vector<DirichletConstraint> constraints(unknowns);
  for (const auto *node : nodes) {
    if (node == nullptr || node->Nodal_Vars_Info == nullptr ||
        node->Nodal_Vars_Info[0] == nullptr || node->First_Unknown == nullptr) {
      throw std::invalid_argument("Missing legacy nodal unknown metadata");
    }
    const int count = node->Nodal_Vars_Info[0]->Num_Unknowns;
    const int first = node->First_Unknown[0];
    if (count < 0 ||
        (count > 0 && (first < 0 || static_cast<std::size_t>(first) + count > unknowns))) {
      throw std::invalid_argument("Legacy nodal unknown range is outside the solution vector");
    }
    if (node->DBC == nullptr || node->DBC[0] == nullptr) {
      continue;
    }
    for (int offset = 0; offset < count; ++offset) {
      const int bc = node->DBC[0][offset];
      if (bc == -1) {
        continue;
      }
      if (bc < 0 || static_cast<std::size_t>(bc) >= boundaries.size() ||
          !constant_boundary(boundaries[bc], 3)) {
        throw std::invalid_argument("Unsupported or invalid nodal Dirichlet reference");
      }
      const auto &boundary = boundaries[bc];
      const DirichletConstraint next{boundary.BC_relax == -1.0 ? DirichletMode::hard_set
                                                               : DirichletMode::residual,
                                     boundary.BC_Data_Float[0]};
      auto &current = constraints[first + offset];
      if (current.mode != DirichletMode::none &&
          (current.mode != next.mode || current.value != next.value)) {
        throw std::invalid_argument("Conflicting nodal Dirichlet offsets");
      }
      current = next;
    }
  }
  return constraints;
}

template <int Dim>
HostFlowSnapshot<Dim> snapshot_legacy_flow(const LegacyFlowInput &input,
                                           std::span<const GomaGlobalOrdinal> global_ids,
                                           std::span<const DirichletConstraint> constraints,
                                           const LegacyDofResolver &resolve) {
  using Topology = Q2Q1<Dim>;
  if (!check_legacy_flow(input).empty() || input.mesh.num_dim != Dim || !resolve ||
      global_ids.size() != constraints.size() || global_ids.size() > INT_MAX) {
    throw std::invalid_argument("Unsupported or inconsistent legacy flow snapshot input");
  }
  HostFlowSnapshot<Dim> result;
  result.unknowns = global_ids.size();
  const auto &mp = *input.materials[0];
  const auto &pd = *input.problems[0];
  result.material.density = mp.density;
  result.material.viscosity = mp.viscosity;
  result.expressions.transient = pd.TimeIntegration == TRANSIENT;
  for (int a = 0; a < Dim; ++a) {
    result.material.source[a] = mp.momentum_source[a];
    const int equation = R_MOMENTUM1 + a;
    const int flags = pd.e[0][equation];
    auto &terms = result.expressions.momentum[a];
    terms.mass = {(flags & T_MASS) != 0, pd.etm[0][equation][LOG2_MASS]};
    terms.advection = {(flags & T_ADVECTION) != 0, pd.etm[0][equation][LOG2_ADVECTION]};
    terms.stress = {(flags & T_DIFFUSION) != 0, pd.etm[0][equation][LOG2_DIFFUSION]};
    terms.source = {(flags & T_SOURCE) != 0, pd.etm[0][equation][LOG2_SOURCE]};
  }
  result.expressions.continuity = {(pd.e[0][R_PRESSURE] & T_ADVECTION) != 0,
                                   pd.etm[0][R_PRESSURE][LOG2_ADVECTION]};
  const auto &mesh = input.mesh;
  if (mesh.num_nodes < 0 || (mesh.num_elem_blocks > 0 && mesh.eb_num_elems == nullptr) ||
      (mesh.num_nodes > 0 && (mesh.x_coord == nullptr || mesh.y_coord == nullptr ||
                              (Dim == 3 && mesh.z_coord == nullptr)))) {
    throw std::invalid_argument("Missing legacy mesh coordinates or connectivity");
  }
  for (int block = 0; block < mesh.num_elem_blocks; ++block) {
    if (mesh.eb_num_elems[block] < 0 ||
        (mesh.eb_num_elems[block] > 0 &&
         (mesh.eb_conn == nullptr || mesh.eb_conn[block] == nullptr))) {
      throw std::invalid_argument("Missing legacy element connectivity");
    }
    for (int e = 0; e < mesh.eb_num_elems[block]; ++e) {
      if (++result.elements > INT_MAX) {
        throw std::invalid_argument("Too many legacy flow elements");
      }
      const int *nodes =
          mesh.eb_conn[block] + static_cast<std::size_t>(e) * Topology::velocity_nodes;
      for (int node = 0; node < Topology::velocity_nodes; ++node) {
        if (nodes[node] < 0 || nodes[node] >= mesh.num_nodes) {
          throw std::invalid_argument("Legacy element references a node outside the mesh");
        }
        for (int d = 0; d < Dim; ++d) {
          const double coordinate =
              (d == 0 ? mesh.x_coord : (d == 1 ? mesh.y_coord : mesh.z_coord))[nodes[node]];
          if (!std::isfinite(coordinate)) {
            throw std::invalid_argument("Nonfinite legacy mesh coordinate");
          }
          result.coordinates.push_back(coordinate);
        }
      }
      std::set<int> used;
      std::set<GomaGlobalOrdinal> used_global;
      for (int row = 0; row < Topology::dofs; ++row) {
        const bool velocity = row < Topology::velocity_dofs;
        const int node =
            nodes[velocity ? row % Topology::velocity_nodes : row - Topology::velocity_dofs];
        const int variable = velocity ? VELOCITY1 + row / Topology::velocity_nodes : PRESSURE;
        const int index = resolve(node, variable, input.block_materials[block]);
        if (index < 0 || static_cast<std::size_t>(index) >= global_ids.size() ||
            !used.insert(index).second || !used_global.insert(global_ids[index]).second) {
          throw std::invalid_argument("Missing, duplicate, or invalid legacy flow DOF");
        }
        result.local_dofs.push_back(index);
        result.global_dofs.push_back(global_ids[index]);
        result.constraints.push_back(constraints[index]);
      }
    }
  }
  if (result.elements != static_cast<std::size_t>(mesh.num_elems)) {
    throw std::invalid_argument("Legacy block counts do not cover the mesh elements");
  }
  return result;
}

template <int Dim>
HostFlowSnapshot<Dim> snapshot_initialized_legacy_flow(const LegacyFlowInput &input,
                                                       GomaSparseMatrix matrix) {
  if (matrix == nullptr || matrix->n_cols < 0 || matrix->n_cols > INT_MAX || matrix->n_rows < 0 ||
      matrix->n_rows > matrix->n_cols || (matrix->n_cols > 0 && matrix->global_ids == nullptr) ||
      input.mesh.num_nodes < 0 || (input.mesh.num_nodes > 0 && Nodes == nullptr)) {
    throw std::invalid_argument("Missing or invalid initialized legacy matrix/node map");
  }
  const auto unknowns = static_cast<std::size_t>(matrix->n_cols);
  const auto constraints = legacy_flow_constraints(
      {Nodes, static_cast<std::size_t>(input.mesh.num_nodes)}, input.boundaries, unknowns);
  // Validate descriptor storage before Index_Solution dereferences it. This
  // adapter accepts one scalar DOF per variable, with material-specific or shared IDs.
  for (int n = 0; n < input.mesh.num_nodes; ++n) {
    const auto &vars = *Nodes[n]->Nodal_Vars_Info[0];
    if (vars.Num_Var_Desc < 0 || (vars.Num_Var_Desc > 0 && (vars.Var_Desc_List == nullptr ||
                                                            vars.Nodal_Offset == nullptr))) {
      throw std::invalid_argument("Missing legacy variable descriptors");
    }
    for (int v = 0; v < vars.Num_Var_Desc; ++v) {
      const auto *description = vars.Var_Desc_List[v];
      if (description == nullptr || description->Ndof != 1 || description->Variable_Type < 0 ||
          description->Variable_Type >= V_LAST || vars.Nodal_Offset[v] < 0 ||
          vars.Nodal_Offset[v] >= vars.Num_Unknowns) {
        throw std::invalid_argument("Invalid or unsupported legacy scalar variable descriptor");
      }
    }
  }
  return snapshot_legacy_flow<Dim>(input, {matrix->global_ids, unknowns}, constraints,
                                   [](int node, int variable, int material) {
                                     return Index_Solution(node, variable, 0, 0, material, 0);
                                   });
}

namespace {
bool all_ranks(bool valid) {
  int local = valid ? 1 : 0, global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, goma_comm);
  return global != 0;
}

// The Tpetra row map is built from the leading n_rows global IDs, so Goma's owned
// residual prefix is already in row-map order. Verify that rather than reorder.
void require_owned_row_map(GomaSparseMatrix matrix) {
  if (matrix == nullptr || matrix->type != GOMA_SPARSE_MATRIX_TYPE_TPETRA ||
      matrix->data == nullptr) {
    throw std::invalid_argument("A configured Tpetra matrix is required");
  }
  const auto &adapter = *static_cast<const TpetraSparseMatrix *>(matrix->data);
  if (adapter.row_map.is_null() || adapter.matrix.is_null() || matrix->n_rows < 0 ||
      adapter.row_map->getLocalNumElements() != static_cast<std::size_t>(matrix->n_rows) ||
      (matrix->n_rows > 0 && matrix->global_ids == nullptr)) {
    throw std::invalid_argument("Tpetra owned rows do not match the legacy matrix extents");
  }
  for (GomaGlobalOrdinal row = 0; row < matrix->n_rows; ++row) {
    if (adapter.row_map->getGlobalElement(static_cast<LO>(row)) != matrix->global_ids[row]) {
      throw std::invalid_argument("Tpetra owned row map differs from the host vector map");
    }
  }
}

template <int Dim> void require_unique_elements(const HostFlowSnapshot<Dim> &snapshot) {
  using Topology = Q2Q1<Dim>;
  std::set<std::vector<int>> elements;
  for (std::size_t e = 0; e < snapshot.elements; ++e) {
    const auto first = snapshot.local_dofs.begin() + e * Topology::dofs;
    std::vector<int> dofs(first, first + Topology::dofs);
    std::sort(dofs.begin(), dofs.end());
    if (!elements.insert(std::move(dofs)).second) {
      throw std::invalid_argument("Legacy mesh includes an element more than once");
    }
  }
}

template <int Dim>
bool owned_pressure_constraint(const HostFlowSnapshot<Dim> &snapshot, std::size_t owned_rows) {
  using Topology = Q2Q1<Dim>;
  for (std::size_t e = 0; e < snapshot.elements; ++e) {
    for (int row = Topology::velocity_dofs; row < Topology::dofs; ++row) {
      const auto offset = e * Topology::dofs + row;
      if (snapshot.constraints[offset].mode != DirichletMode::none &&
          static_cast<std::size_t>(snapshot.local_dofs[offset]) < owned_rows) {
        return true;
      }
    }
  }
  return false;
}

template <int Dim>
void setup_legacy_dimension(goma_kokkos_context *context,
                            const LegacyFlowInput &input,
                            GomaSparseMatrix matrix) {
  std::string error;
  HostFlowSnapshot<Dim> snapshot;
  try {
    if (context == nullptr) {
      throw std::invalid_argument("Missing Kokkos assembly context");
    }
    const auto issues = check_legacy_flow(input);
    if (!issues.empty()) {
      throw std::invalid_argument("Unsupported Kokkos flow configuration: " + issues.front());
    }
    require_owned_row_map(matrix);
    snapshot = snapshot_initialized_legacy_flow<Dim>(input, matrix);
    require_unique_elements(snapshot);
  } catch (const std::exception &failure) {
    error = failure.what();
  } catch (...) {
    error = "Unknown legacy snapshot failure";
  }
  // Join local failures before any rank enters collective context setup.
  if (!all_ranks(error.empty())) {
    throw std::invalid_argument(error.empty() ? "Kokkos legacy setup failed on another rank"
                                              : error);
  }
  int local = owned_pressure_constraint(snapshot, static_cast<std::size_t>(matrix->n_rows)) ? 1 : 0;
  int pressure = 0;
  MPI_Allreduce(&local, &pressure, 1, MPI_INT, MPI_MAX, goma_comm);
  if (pressure == 0) {
    throw std::invalid_argument("No rank owns a pressure DOF with a nodal P constraint");
  }
  setup_context<Dim>(context, snapshot, matrix, DIRICHLET_PENALTY);
}
} // namespace

void setup_legacy_context(goma_kokkos_context *context,
                          const LegacyFlowInput &input,
                          GomaSparseMatrix matrix) {
  try {
    const int dimension = input.mesh.num_dim;
    int minimum = 0, maximum = 0;
    MPI_Allreduce(&dimension, &minimum, 1, MPI_INT, MPI_MIN, goma_comm);
    MPI_Allreduce(&dimension, &maximum, 1, MPI_INT, MPI_MAX, goma_comm);
    if (minimum != maximum || (minimum != 2 && minimum != 3)) {
      throw std::invalid_argument("Kokkos legacy mesh dimensions are unsupported or differ");
    }
    if (dimension == 2) {
      setup_legacy_dimension<2>(context, input, matrix);
    } else {
      setup_legacy_dimension<3>(context, input, matrix);
    }
  } catch (const std::exception &error) {
    detail::context_error(context, error.what());
    throw;
  }
}

template HostFlowSnapshot<2> snapshot_initialized_legacy_flow<2>(const LegacyFlowInput &,
                                                                 GomaSparseMatrix);
template HostFlowSnapshot<3> snapshot_initialized_legacy_flow<3>(const LegacyFlowInput &,
                                                                 GomaSparseMatrix);

template HostFlowSnapshot<2> snapshot_legacy_flow<2>(const LegacyFlowInput &,
                                                     std::span<const GomaGlobalOrdinal>,
                                                     std::span<const DirichletConstraint>,
                                                     const LegacyDofResolver &);
template HostFlowSnapshot<3> snapshot_legacy_flow<3>(const LegacyFlowInput &,
                                                     std::span<const GomaGlobalOrdinal>,
                                                     std::span<const DirichletConstraint>,
                                                     const LegacyDofResolver &);
} // namespace goma::assembly

extern "C" goma_kokkos_status goma_kokkos_preflight_legacy(const Exo_DB *exo) {
  try {
    using namespace goma::assembly;
    if (exo == nullptr || upd == nullptr) {
      return GOMA_KOKKOS_INVALID_ARGUMENT;
    }
    std::vector<std::string> issues;
    try {
      const auto input = describe_legacy_flow(*exo);
      issues = check_legacy_flow(input);
    } catch (const std::exception &error) {
      issues.push_back(std::string("adapter: preflight failed: ") + error.what());
    } catch (...) {
      issues.push_back("adapter: preflight failed with an unknown exception");
    }
    // Never let an eligible subset run through legacy assembly while routing is unfinished.
    issues.push_back("execution: Kokkos solver routing and complete legacy assembly parity are not "
                     "yet available");
    try {
      int rank = 0, processes = 1;
      MPI_Comm_rank(goma_comm, &rank);
      MPI_Comm_size(goma_comm, &processes);
      std::string local;
      for (const auto &issue : issues) {
        local += issue + '\n';
      }
      const int length = static_cast<int>(local.size());
      std::vector<int> counts(rank == 0 ? processes : 0);
      MPI_Gather(&length, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, goma_comm);
      std::vector<int> offsets(rank == 0 ? processes : 0);
      std::size_t total = 0;
      int valid = 1;
      if (rank == 0) {
        for (int p = 0; p < processes; ++p) {
          if (total > INT_MAX) {
            valid = 0;
            break;
          }
          offsets[p] = static_cast<int>(total);
          total += counts[p];
        }
        valid = valid && total <= INT_MAX;
      }
      MPI_Bcast(&valid, 1, MPI_INT, 0, goma_comm);
      if (!valid) {
        return GOMA_KOKKOS_RUNTIME_ERROR;
      }
      std::vector<char> gathered(rank == 0 ? total : 0);
      MPI_Gatherv(local.data(), length, MPI_CHAR, gathered.data(), counts.data(), offsets.data(),
                  MPI_CHAR, 0, goma_comm);
      if (rank == 0) {
        std::set<std::string> unique;
        std::istringstream lines(std::string(gathered.begin(), gathered.end()));
        std::string line;
        while (std::getline(lines, line)) {
          unique.insert(line);
        }
        std::fprintf(stderr, "Kokkos assembly preflight rejected this configuration:\n");
        for (const auto &issue : unique) {
          std::fprintf(stderr, "  - %s\n", issue.c_str());
        }
        std::fflush(stderr);
      }
    } catch (...) {
      return GOMA_KOKKOS_RUNTIME_ERROR;
    }
    return GOMA_KOKKOS_UNSUPPORTED;
  } catch (...) {
    return GOMA_KOKKOS_RUNTIME_ERROR;
  }
}

extern "C" goma_kokkos_status goma_kokkos_context_setup_legacy(goma_kokkos_context *context,
                                                               const Exo_DB *exo,
                                                               struct g_GomaSparseMatrix *matrix) {
  using namespace goma::assembly;
  try {
    std::optional<LegacyFlowInput> input;
    std::string error;
    try {
      if (exo == nullptr) {
        throw std::invalid_argument("Missing legacy mesh");
      }
      input.emplace(describe_legacy_flow(*exo));
    } catch (const std::exception &failure) {
      error = failure.what();
    }
    if (!all_ranks(error.empty())) {
      detail::context_error(context, error.empty() ? "Kokkos legacy setup failed on another rank"
                                                   : error.c_str());
      return GOMA_KOKKOS_INVALID_ARGUMENT;
    }
    setup_legacy_context(context, *input, matrix);
    return GOMA_KOKKOS_SUCCESS;
  } catch (const std::invalid_argument &) {
    return GOMA_KOKKOS_INVALID_ARGUMENT;
  } catch (const std::exception &error) {
    detail::context_error(context, error.what());
    return GOMA_KOKKOS_RUNTIME_ERROR;
  } catch (...) {
    detail::context_error(context, "Unknown Kokkos legacy setup failure");
    return GOMA_KOKKOS_RUNTIME_ERROR;
  }
}

extern "C" goma_kokkos_status
goma_kokkos_context_assemble_legacy(goma_kokkos_context *context,
                                    struct g_GomaSparseMatrix *matrix,
                                    const double *x,
                                    const double *xdot,
                                    double time_coefficient,
                                    goma_kokkos_evaluation_request request,
                                    double *resid_vector) {
  using goma::assembly::all_ranks;
  try {
    // Reject a different matrix collectively; its extents would desynchronize ranks.
    const bool valid = context != nullptr && matrix != nullptr && context->matrix == matrix &&
                       matrix->n_rows >= 0 && matrix->n_cols >= matrix->n_rows;
    if (!all_ranks(valid)) {
      goma::assembly::detail::context_error(
          context, "Kokkos legacy assembly requires the configured context and setup matrix");
      return GOMA_KOKKOS_INVALID_ARGUMENT;
    }
    const auto unknowns = static_cast<size_t>(matrix->n_cols);
    return goma_kokkos_context_assemble(context, x, unknowns, xdot, xdot == nullptr ? 0 : unknowns,
                                        time_coefficient, request, resid_vector,
                                        static_cast<size_t>(matrix->n_rows));
  } catch (...) {
    goma::assembly::detail::context_error(context, "Unknown Kokkos legacy assembly failure");
    return GOMA_KOKKOS_RUNTIME_ERROR;
  }
}
