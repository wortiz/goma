/* Fixed-mesh Q2/Q1 element evaluation; no constraints or global scatter yet. */
#ifndef GOMA_MM_KOKKOS_ELEMENT_HPP
#define GOMA_MM_KOKKOS_ELEMENT_HPP

#include "mm_kokkos_basis.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace goma::assembly {

enum class ElementStatus {
  success = 0,
  nonfinite_geometry = 1,
  nonpositive_jacobian = 2,
  invalid_input = 3,
  nonfinite_contribution = 4
};

enum class EvaluationRequest { residual = 1, jacobian = 2, both = 3 };

template <int Dim> struct QuadratureGeometry {
  Basis<Dim> velocity[Q2Q1<Dim>::velocity_nodes];
  Basis<Dim> pressure[Q2Q1<Dim>::pressure_nodes];
  double position[Dim]{};
  double determinant = 0.0;
  double weight = 0.0; // Includes the geometry determinant.
};

template <int Dim> struct ElementGeometry {
  Kokkos::View<QuadratureGeometry<Dim> **> points;
  Kokkos::View<int *> status;
};

using ElementCoordinates = Kokkos::View<double ***>;
using ElementFields = Kokkos::View<double **>;

template <int Dim> struct ElementResult {
  Kokkos::View<double **> residual;
  Kokkos::View<double ***> jacobian;
  Kokkos::View<int *> status;
};

namespace detail {

template <int Dim>
KOKKOS_INLINE_FUNCTION ElementStatus map_point(const ElementCoordinates &coordinates,
                                               int element,
                                               int point,
                                               QuadratureGeometry<Dim> &result) {
  using Topology = Q2Q1<Dim>;
  double xi[Dim];
  const double weight = Topology::quadrature(point, xi);
  double jacobian[Dim][Dim]{}; // [physical direction][reference direction]
  for (int node = 0; node < Topology::velocity_nodes; ++node) {
    const auto basis = Topology::template basis<2>(node, xi);
    for (int d = 0; d < Dim; ++d) {
      const double coordinate = coordinates(element, node, d);
      if (!Kokkos::isfinite(coordinate)) {
        return ElementStatus::nonfinite_geometry;
      }
      result.position[d] += coordinate * basis.value;
      for (int r = 0; r < Dim; ++r) {
        jacobian[d][r] += coordinate * basis.gradient[r];
      }
    }
  }
  double inverse[Dim][Dim]; // [reference direction][physical direction]
  double determinant;
  if constexpr (Dim == 2) {
    determinant = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
    inverse[0][0] = jacobian[1][1];
    inverse[0][1] = -jacobian[0][1];
    inverse[1][0] = -jacobian[1][0];
    inverse[1][1] = jacobian[0][0];
  } else {
    for (int r = 0; r < Dim; ++r) {
      for (int d = 0; d < Dim; ++d) {
        inverse[r][d] = jacobian[(d + 1) % 3][(r + 1) % 3] * jacobian[(d + 2) % 3][(r + 2) % 3] -
                        jacobian[(d + 1) % 3][(r + 2) % 3] * jacobian[(d + 2) % 3][(r + 1) % 3];
      }
    }
    determinant = jacobian[0][0] * inverse[0][0] + jacobian[0][1] * inverse[1][0] +
                  jacobian[0][2] * inverse[2][0];
  }
  if (!Kokkos::isfinite(determinant)) {
    return ElementStatus::nonfinite_geometry;
  }
  if (determinant <= 0.0) {
    return ElementStatus::nonpositive_jacobian;
  }
  result.determinant = determinant;
  result.weight = weight * determinant;
  for (int d = 0; d < Dim; ++d) {
    if (!Kokkos::isfinite(result.position[d])) {
      return ElementStatus::nonfinite_geometry;
    }
    for (int r = 0; r < Dim; ++r) {
      inverse[r][d] /= determinant;
      if (!Kokkos::isfinite(inverse[r][d])) {
        return ElementStatus::nonfinite_geometry;
      }
    }
  }
  for (int family = 0; family < 2; ++family) {
    const int nodes = family == 0 ? Topology::velocity_nodes : Topology::pressure_nodes;
    for (int node = 0; node < nodes; ++node) {
      const auto reference = family == 0 ? Topology::template basis<2>(node, xi)
                                         : Topology::template basis<1>(node, xi);
      auto &basis = family == 0 ? result.velocity[node] : result.pressure[node];
      basis.value = reference.value;
      for (int d = 0; d < Dim; ++d) {
        basis.gradient[d] = 0.0;
        for (int r = 0; r < Dim; ++r) {
          basis.gradient[d] += reference.gradient[r] * inverse[r][d];
        }
        if (!Kokkos::isfinite(basis.gradient[d])) {
          return ElementStatus::nonfinite_geometry;
        }
      }
    }
  }
  return ElementStatus::success;
}

template <int Dim>
KOKKOS_INLINE_FUNCTION FlowState<double, Dim> interpolate(const QuadratureGeometry<Dim> &point,
                                                          const ElementFields &solution,
                                                          const ElementFields &velocity_dot,
                                                          int element,
                                                          bool transient) {
  using Topology = Q2Q1<Dim>;
  FlowState<double, Dim> state;
  for (int a = 0; a < Dim; ++a) {
    for (int node = 0; node < Topology::velocity_nodes; ++node) {
      const int dof = a * Topology::velocity_nodes + node;
      const double value = solution(element, dof);
      state.velocity[a] += value * point.velocity[node].value;
      if (transient) {
        state.velocity_dot[a] += velocity_dot(element, dof) * point.velocity[node].value;
      }
      for (int d = 0; d < Dim; ++d) {
        state.gradient[d][a] += value * point.velocity[node].gradient[d];
      }
    }
  }
  for (int node = 0; node < Topology::pressure_nodes; ++node) {
    state.pressure +=
        solution(element, Topology::velocity_dofs + node) * point.pressure[node].value;
  }
  return state;
}

template <int Dim>
KOKKOS_INLINE_FUNCTION double row_residual(const FlowExpressions<Dim> &flow,
                                           const FlowState<double, Dim> &state,
                                           const MaterialValues<double, Dim> &material,
                                           const QuadratureGeometry<Dim> &point,
                                           int row) {
  using Topology = Q2Q1<Dim>;
  if (row < Topology::velocity_dofs) {
    return flow.momentum_residual(state, material, point.velocity[row % Topology::velocity_nodes],
                                  row / Topology::velocity_nodes, point.weight);
  }
  return flow.continuity_residual(state, point.pressure[row - Topology::velocity_dofs],
                                  point.weight);
}

template <int Dim>
KOKKOS_INLINE_FUNCTION bool valid_configuration(const ConstantFluid<Dim> &fluid,
                                                const FlowExpressions<Dim> &flow) {
  if (!(fluid.density > 0.0) || !(fluid.viscosity >= 0.0) || !Kokkos::isfinite(fluid.density) ||
      !Kokkos::isfinite(fluid.viscosity)) {
    return false;
  }
  for (int a = 0; a < Dim; ++a) {
    if (!Kokkos::isfinite(fluid.source[a])) {
      return false;
    }
    const auto &terms = flow.momentum[a];
    const Term entries[] = {terms.mass, terms.advection, terms.stress, terms.source};
    for (const auto &term : entries) {
      if (term.enabled && !Kokkos::isfinite(term.multiplier)) {
        return false;
      }
    }
  }
  return !flow.continuity.enabled || Kokkos::isfinite(flow.continuity.multiplier);
}

template <int Dim, bool Residual, bool Jacobian>
void launch_elements(const ElementGeometry<Dim> &geometry,
                     const ElementFields &solution,
                     const ElementFields &velocity_dot,
                     const Kokkos::View<ConstantFluid<Dim> *> &materials,
                     const Kokkos::View<FlowExpressions<Dim> *> &expressions,
                     double time_coefficient,
                     const ElementResult<Dim> &output) {
  using Topology = Q2Q1<Dim>;
  using Policy = Kokkos::TeamPolicy<>;
  using Team = typename Policy::member_type;
  using Scratch = Kokkos::View<FlowState<double, Dim> *, typename Team::scratch_memory_space,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Policy policy(static_cast<int>(solution.extent(0)), Kokkos::AUTO);
  policy.set_scratch_size(0, Kokkos::PerTeam(Scratch::shmem_size(Topology::quadrature_points)));
  Kokkos::parallel_for(
      "goma_q2q1_elements", policy, KOKKOS_LAMBDA(const Team &team) {
        const int element = team.league_rank();
        const auto fluid = materials(element);
        const auto flow = expressions(element);
        // nvcc rejects extended lambdas whose first capture is inside if constexpr.
        const double coefficient = time_coefficient;
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          int status = geometry.status(element);
          if (status == 0) {
            bool valid = valid_configuration(fluid, flow);
            for (int dof = 0; dof < Topology::dofs; ++dof) {
              valid = valid && Kokkos::isfinite(solution(element, dof));
            }
            if (flow.transient) {
              for (int dof = 0; dof < Topology::velocity_dofs; ++dof) {
                valid = valid && Kokkos::isfinite(velocity_dot(element, dof));
              }
            }
            if (!valid) {
              status = static_cast<int>(ElementStatus::invalid_input);
            }
          }
          output.status(element) = status;
        });
        team.team_barrier();
        if (output.status(element) != 0) {
          return;
        }
        Scratch states(team.team_scratch(0), Topology::quadrature_points);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, Topology::quadrature_points),
                             [&](int q) {
                               states(q) = interpolate(geometry.points(element, q), solution,
                                                       velocity_dot, element, flow.transient);
                             });
        team.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, Topology::dofs), [&](int row) {
          double residual = 0.0;
          if constexpr (Jacobian) {
            double entries[Topology::dofs]{};
            for (int q = 0; q < Topology::quadrature_points; ++q) {
              const auto &point = geometry.points(element, q);
              const auto &state = states(q);
              const auto material = fluid.evaluate(state);
              if constexpr (Residual) {
                residual += row_residual(flow, state, material, point, row);
              }
              for (int column = 0; column < Topology::dofs; ++column) {
                const auto increment =
                    column < Topology::velocity_dofs
                        ? velocity_increment(point.velocity[column % Topology::velocity_nodes],
                                             column / Topology::velocity_nodes, coefficient)
                        : pressure_increment(point.pressure[column - Topology::velocity_dofs]);
                const auto tangent = fluid.linearize(state, increment);
                entries[column] +=
                    row < Topology::velocity_dofs
                        ? flow.momentum_linearization(
                              state, increment, material, tangent,
                              point.velocity[row % Topology::velocity_nodes],
                              row / Topology::velocity_nodes, point.weight)
                        : flow.continuity_linearization(
                              increment, point.pressure[row - Topology::velocity_dofs],
                              point.weight);
              }
            }
            for (int column = 0; column < Topology::dofs; ++column) {
              if (!Kokkos::isfinite(entries[column])) {
                Kokkos::atomic_max(&output.status(element),
                                   static_cast<int>(ElementStatus::nonfinite_contribution));
              }
              output.jacobian(element, row, column) = entries[column];
            }
          } else {
            for (int q = 0; q < Topology::quadrature_points; ++q) {
              const auto &state = states(q);
              residual += row_residual(flow, state, fluid.evaluate(state),
                                       geometry.points(element, q), row);
            }
          }
          if constexpr (Residual) {
            if (!Kokkos::isfinite(residual)) {
              Kokkos::atomic_max(&output.status(element),
                                 static_cast<int>(ElementStatus::nonfinite_contribution));
            }
            output.residual(element, row) = residual;
          }
        });
      });
}
} // namespace detail

// All launches use the default execution-space instance. These functions return
// asynchronously; inspect status after a fence or a host mirror copy. A nonzero
// status invalidates the entire element's output, including any partial values.
// Keep input/output Views and the runtime alive until the launches complete.
template <int Dim>
ElementGeometry<Dim> prepare_q2q1_geometry(const ElementCoordinates &coordinates) {
  using Topology = Q2Q1<Dim>;
  const auto elements = coordinates.extent(0);
  if (elements > INT_MAX || coordinates.extent(1) != Topology::velocity_nodes ||
      coordinates.extent(2) != Dim) {
    throw std::invalid_argument(
        "Q2/Q1 coordinates must have shape (elements, Q2 nodes, dimension)");
  }
  ElementGeometry<Dim> geometry;
  geometry.points = Kokkos::View<QuadratureGeometry<Dim> **>("goma_q2q1_geometry", elements,
                                                             Topology::quadrature_points);
  geometry.status = Kokkos::View<int *>("goma_q2q1_geometry_status", elements);
  using Policy = Kokkos::TeamPolicy<>;
  using Team = typename Policy::member_type;
  Kokkos::parallel_for(
      "goma_q2q1_geometry", Policy(static_cast<int>(elements), Kokkos::AUTO),
      KOKKOS_LAMBDA(const Team &team) {
        const int element = team.league_rank();
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, Topology::quadrature_points), [&](int q) {
              const auto status =
                  detail::map_point(coordinates, element, q, geometry.points(element, q));
              Kokkos::atomic_max(&geometry.status(element), static_cast<int>(status));
            });
      });
  return geometry;
}

template <int Dim>
ElementResult<Dim> evaluate_q2q1_elements(const ElementGeometry<Dim> &geometry,
                                          const ElementFields &solution,
                                          const ElementFields &velocity_dot,
                                          const Kokkos::View<ConstantFluid<Dim> *> &materials,
                                          const Kokkos::View<FlowExpressions<Dim> *> &expressions,
                                          double time_coefficient,
                                          EvaluationRequest request) {
  using Topology = Q2Q1<Dim>;
  const auto elements = solution.extent(0);
  if (elements > INT_MAX || solution.extent(1) != Topology::dofs ||
      velocity_dot.extent(0) != elements || velocity_dot.extent(1) != Topology::velocity_dofs ||
      geometry.points.extent(0) != elements ||
      geometry.points.extent(1) != Topology::quadrature_points ||
      geometry.status.extent(0) != elements || materials.extent(0) != elements ||
      expressions.extent(0) != elements || !std::isfinite(time_coefficient)) {
    throw std::invalid_argument("Inconsistent Q2/Q1 element evaluation inputs");
  }
  if (request != EvaluationRequest::residual && request != EvaluationRequest::jacobian &&
      request != EvaluationRequest::both) {
    throw std::invalid_argument("Invalid element evaluation request");
  }
  ElementResult<Dim> result;
  result.status = Kokkos::View<int *>("goma_q2q1_element_status", elements);
  if (request != EvaluationRequest::jacobian) {
    result.residual =
        Kokkos::View<double **>("goma_q2q1_element_residual", elements, Topology::dofs);
  }
  if (request != EvaluationRequest::residual) {
    result.jacobian = Kokkos::View<double ***>("goma_q2q1_element_jacobian", elements,
                                               Topology::dofs, Topology::dofs);
  }
  if (request == EvaluationRequest::both) {
    detail::launch_elements<Dim, true, true>(geometry, solution, velocity_dot, materials,
                                             expressions, time_coefficient, result);
  } else if (request == EvaluationRequest::residual) {
    detail::launch_elements<Dim, true, false>(geometry, solution, velocity_dot, materials,
                                              expressions, time_coefficient, result);
  } else {
    detail::launch_elements<Dim, false, true>(geometry, solution, velocity_dot, materials,
                                              expressions, time_coefficient, result);
  }
  return result;
}

} // namespace goma::assembly
#endif
