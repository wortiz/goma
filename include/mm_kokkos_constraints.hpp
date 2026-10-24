/* Element-row replacement for constant velocity and pressure Dirichlet data. */
#ifndef GOMA_MM_KOKKOS_CONSTRAINTS_HPP
#define GOMA_MM_KOKKOS_CONSTRAINTS_HPP

#include "mm_kokkos_element.hpp"

namespace goma::assembly {

enum class DirichletMode { none = 0, hard_set = 1, residual = 2 };
struct DirichletConstraint {
  DirichletMode mode = DirichletMode::none;
  double value = 0.0;
};
using ElementConstraints = Kokkos::View<DirichletConstraint **>;

// Supply DIRICHLET_PENALTY from the host adapter. Each incident element receives
// the full penalty. Columns are retained, as in put_dirichlet_in_matrix(). The
// host must impose hard-set solution values and zero their time derivatives
// before gathering fields; row replacement itself never changes those fields.
template <int Dim>
void apply_q2q1_dirichlet(const ElementResult<Dim> &result,
                          const ElementFields &solution,
                          const ElementConstraints &constraints,
                          EvaluationRequest request,
                          double penalty) {
  using Topology = Q2Q1<Dim>;
  const auto elements = solution.extent(0);
  const bool residual =
      request == EvaluationRequest::residual || request == EvaluationRequest::both;
  const bool jacobian =
      request == EvaluationRequest::jacobian || request == EvaluationRequest::both;
  if ((!residual && !jacobian) || elements > INT_MAX || solution.extent(1) != Topology::dofs ||
      constraints.extent(0) != elements || constraints.extent(1) != Topology::dofs ||
      result.status.extent(0) != elements || !std::isfinite(penalty) || penalty <= 0.0 ||
      (residual &&
       (result.residual.extent(0) != elements || result.residual.extent(1) != Topology::dofs)) ||
      (jacobian &&
       (result.jacobian.extent(0) != elements || result.jacobian.extent(1) != Topology::dofs ||
        result.jacobian.extent(2) != Topology::dofs))) {
    throw std::invalid_argument("Inconsistent Q2/Q1 Dirichlet inputs");
  }
  using Policy = Kokkos::TeamPolicy<>;
  using Team = typename Policy::member_type;
  Kokkos::parallel_for(
      "goma_q2q1_dirichlet", Policy(static_cast<int>(elements), Kokkos::AUTO),
      KOKKOS_LAMBDA(const Team &team) {
        const int element = team.league_rank();
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          if (result.status(element) != 0) {
            return;
          }
          for (int row = 0; row < Topology::dofs; ++row) {
            const auto constraint = constraints(element, row);
            if (constraint.mode == DirichletMode::none) {
              continue;
            }
            if ((constraint.mode != DirichletMode::hard_set &&
                 constraint.mode != DirichletMode::residual) ||
                !Kokkos::isfinite(constraint.value) ||
                (residual && constraint.mode == DirichletMode::residual &&
                 !Kokkos::isfinite(solution(element, row)))) {
              result.status(element) = static_cast<int>(ElementStatus::invalid_input);
              return;
            }
          }
        });
        team.team_barrier();
        if (result.status(element) != 0) {
          return;
        }
        // All threads must pass the status check before rows can report overflow.
        team.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, Topology::dofs), [&](int row) {
          const auto constraint = constraints(element, row);
          if (constraint.mode == DirichletMode::none) {
            return;
          }
          if (jacobian) {
            for (int column = 0; column < Topology::dofs; ++column) {
              result.jacobian(element, row, column) = column == row ? penalty : 0.0;
            }
          }
          if (residual) {
            const double value = constraint.mode == DirichletMode::hard_set
                                     ? 0.0
                                     : penalty * (solution(element, row) - constraint.value);
            result.residual(element, row) = value;
            if (!Kokkos::isfinite(value)) {
              Kokkos::atomic_max(&result.status(element),
                                 static_cast<int>(ElementStatus::nonfinite_contribution));
            }
          }
        });
      });
}

} // namespace goma::assembly
#endif
