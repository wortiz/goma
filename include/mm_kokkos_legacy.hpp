/* Host-only adaptation of initialized Goma mesh, material, and nodal metadata. */
#ifndef GOMA_MM_KOKKOS_LEGACY_HPP
#define GOMA_MM_KOKKOS_LEGACY_HPP

#include "mm_kokkos_assembly.h"
#include "mm_kokkos_host.hpp"
extern "C" {
#include "mm_as_structs.h"
#include "mm_mp_structs.h"
}
#include <functional>
#include <string>

namespace goma::assembly {

struct LegacyFlowOptions {
  std::string matrix_format;
  int linear_solver = -1;
  bool pspg = false;
  bool continuity_gls = false;
  bool level_set = false;
  bool xfem = false;
  bool augmenting = false;
  bool continuation = false;
  bool linear_stability = false;
  bool particles = false;
  bool rotations = false;
  bool element_pressure_datum = false;
  bool numerical_jacobian = false;
};

struct LegacyFlowInput {
  const Exo_DB &mesh;
  const UPD_STRUCT &update;
  std::span<PROBLEM_DESCRIPTION_STRUCT *const> problems;
  std::span<MATRL_PROP_STRUCT *const> materials;
  std::span<GEN_NEWT_STRUCT *const> constitutive;
  std::span<const Boundary_Condition> boundaries;
  std::span<const int> block_materials;
  LegacyFlowOptions options;
};

// Read initialized process-global metadata without changing it. Returned spans
// borrow Goma storage; consume them before teardown or another problem setup.
LegacyFlowInput describe_legacy_flow(const Exo_DB &mesh);

// Copy actual owned/ghost global IDs and Nodes DBC metadata, resolving matrix-zero
// unknowns with Index_Solution. Host-only: caller must coordinate failures across
// ranks before entering collective context setup.
template <int Dim>
HostFlowSnapshot<Dim> snapshot_initialized_legacy_flow(const LegacyFlowInput &input,
                                                       GomaSparseMatrix matrix);

// Collective on goma_comm, which must match the matrix communicator. Snapshot the
// initialized metadata, then require on every rank that the Tpetra owned row map
// equals the leading matrix global IDs, that each local element (including ghost
// elements) appears once, and that some rank owns a constrained pressure DOF.
// Sets up the context with DIRICHLET_PENALTY. Throws on every rank if any rank fails.
void setup_legacy_context(goma_kokkos_context *context,
                          const LegacyFlowInput &input,
                          GomaSparseMatrix matrix);

// Collect every known incompatibility instead of stopping at the first one.
// Call after setup_problem(), which resolves the integration-map interpolation.
std::vector<std::string> check_legacy_flow(const LegacyFlowInput &input);

// Build constraints from the existing node DBC offsets, without imposing values.
std::vector<DirichletConstraint>
legacy_flow_constraints(std::span<NODE_INFO_STRUCT *const> nodes,
                        std::span<const Boundary_Condition> boundaries,
                        std::size_t unknowns);

using LegacyDofResolver = std::function<int(int node, int variable, int material)>;
// The resolver is host-only (Index_Solution in the solver adapter). Snapshot owns
// all copied data, and device gathering uses only the resulting integer map.
template <int Dim>
HostFlowSnapshot<Dim> snapshot_legacy_flow(const LegacyFlowInput &input,
                                           std::span<const GomaGlobalOrdinal> global_ids,
                                           std::span<const DirichletConstraint> constraints,
                                           const LegacyDofResolver &resolve);

} // namespace goma::assembly
#endif
