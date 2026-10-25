/* Owned host snapshots and bulk gathering for the legacy-to-device boundary. */
#ifndef GOMA_MM_KOKKOS_HOST_HPP
#define GOMA_MM_KOKKOS_HOST_HPP

#include "linalg/sparse_matrix.h"
#include "mm_kokkos_constraints.hpp"
#include <span>
#include <vector>

namespace goma::assembly {

template <int Dim> struct HostFlowSnapshot {
  std::size_t elements = 0;
  std::size_t unknowns = 0;        // Includes ghost entries of the host solution vector.
  std::vector<double> coordinates; // [element][Q2 node][direction]
  std::vector<int> local_dofs;     // Component-major velocity, then corner pressure.
  std::vector<GomaGlobalOrdinal> global_dofs;
  std::vector<DirichletConstraint> constraints;
  ConstantFluid<Dim> material;
  FlowExpressions<Dim> expressions;
};

template <int Dim> struct GatheredFlowFields {
  ElementFields solution;
  ElementFields velocity_dot;
};

template <int Dim> class HostFieldMap {
public:
  HostFieldMap(std::span<const int> indices, std::size_t elements, std::size_t unknowns)
      : unknowns_(unknowns) {
    using Topology = Q2Q1<Dim>;
    if (elements > INT_MAX || unknowns > INT_MAX || indices.size() != elements * Topology::dofs) {
      throw std::invalid_argument("Invalid host flow DOF-map extents");
    }
    for (int index : indices) {
      if (index < 0 || static_cast<std::size_t>(index) >= unknowns) {
        throw std::invalid_argument("Host flow DOF index is outside the local solution vector");
      }
    }
    indices_ = Kokkos::View<int **>("goma_host_flow_indices", elements, Topology::dofs);
    auto host = Kokkos::create_mirror_view(indices_);
    for (std::size_t e = 0; e < elements; ++e) {
      for (int dof = 0; dof < Topology::dofs; ++dof) {
        host(e, dof) = indices[e * Topology::dofs + dof];
      }
    }
    Kokkos::deep_copy(indices_, host);
  }

  // The caller exchanges ghosts and applies hard-set values before this call.
  // Host spans are copied in bulk before returning; kernels never access host pointers.
  GatheredFlowFields<Dim> gather(std::span<const double> solution,
                                 std::span<const double> velocity_dot,
                                 bool transient) const {
    using Topology = Q2Q1<Dim>;
    if (solution.size() != unknowns_ || (transient && velocity_dot.size() != unknowns_)) {
      throw std::invalid_argument("Host flow solution vector has the wrong length");
    }
    using HostValues =
        Kokkos::View<const double *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Kokkos::View<double *> values("goma_host_flow_solution", unknowns_);
    Kokkos::deep_copy(values, HostValues(solution.data(), solution.size()));
    Kokkos::View<double *> dot;
    if (transient) {
      dot = Kokkos::View<double *>("goma_host_flow_dot", unknowns_);
      Kokkos::deep_copy(dot, HostValues(velocity_dot.data(), velocity_dot.size()));
    }
    const auto indices = indices_;
    const auto elements = indices.extent(0);
    GatheredFlowFields<Dim> result{
        ElementFields("goma_gathered_solution", elements, Topology::dofs),
        ElementFields("goma_gathered_dot", elements, Topology::velocity_dofs)};
    // One iteration per element avoids overflowing an int range for e*dofs.
    Kokkos::parallel_for(
        "goma_gather_host_flow", static_cast<int>(elements), KOKKOS_LAMBDA(int e) {
          for (int dof = 0; dof < Topology::dofs; ++dof) {
            result.solution(e, dof) = values(indices(e, dof));
            if (dof < Topology::velocity_dofs) {
              result.velocity_dot(e, dof) = transient ? dot(indices(e, dof)) : 0.0;
            }
          }
        });
    // Bulk temporary vectors and host input spans can be released after return.
    Kokkos::fence("Goma host flow gather");
    return result;
  }

private:
  std::size_t unknowns_;
  Kokkos::View<int **> indices_;
};

} // namespace goma::assembly
#endif
