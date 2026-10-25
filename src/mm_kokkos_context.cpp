#include "mm_kokkos_context.hpp"
#include "mm_kokkos_context_internal.hpp"
#include "mm_kokkos_tpetra.hpp"

#include "linalg/sparse_matrix_tpetra.h"
#include <Teuchos_CommHelpers.hpp>
#include <unordered_map>
#include <unordered_set>

namespace goma::assembly {
namespace {
void require_all(const Teuchos::Comm<int> &comm, bool valid, const char *message) {
  const int local = valid ? 0 : 1;
  int global = 0;
  Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &local, &global);
  if (global != 0) {
    throw std::invalid_argument(message);
  }
}

void require_agreement(const Teuchos::Comm<int> &comm, int value, const char *message) {
  int minimum = 0, maximum = 0;
  Teuchos::reduceAll(comm, Teuchos::REDUCE_MIN, 1, &value, &minimum);
  Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &value, &maximum);
  if (minimum != maximum) {
    throw std::invalid_argument(message);
  }
}

template <int Dim>
bool valid_snapshot(const HostFlowSnapshot<Dim> &snapshot,
                    const Tpetra::Map<LO, GO> &owned,
                    double penalty) {
  using Topology = Q2Q1<Dim>;
  if (snapshot.elements > INT_MAX || snapshot.unknowns > INT_MAX || !std::isfinite(penalty) ||
      penalty <= 0.0 ||
      snapshot.coordinates.size() != snapshot.elements * Topology::velocity_nodes * Dim ||
      snapshot.local_dofs.size() != snapshot.elements * Topology::dofs ||
      snapshot.global_dofs.size() != snapshot.local_dofs.size() ||
      snapshot.constraints.size() != snapshot.local_dofs.size() ||
      !detail::valid_configuration(snapshot.material, snapshot.expressions)) {
    return false;
  }
  for (double coordinate : snapshot.coordinates) {
    if (!std::isfinite(coordinate)) {
      return false;
    }
  }
  std::unordered_map<int, GO> local_to_global;
  std::unordered_map<GO, int> global_to_local;
  std::unordered_map<int, int> roles;
  std::unordered_map<int, DirichletConstraint> constraints;
  for (std::size_t e = 0; e < snapshot.elements; ++e) {
    std::unordered_set<int> unique;
    for (int row = 0; row < Topology::dofs; ++row) {
      const auto offset = e * Topology::dofs + row;
      const int local = snapshot.local_dofs[offset];
      const GO global = snapshot.global_dofs[offset];
      const auto constraint = snapshot.constraints[offset];
      if (local < 0 || static_cast<std::size_t>(local) >= snapshot.unknowns ||
          !unique.insert(local).second ||
          (constraint.mode != DirichletMode::none &&
           ((constraint.mode != DirichletMode::hard_set &&
             constraint.mode != DirichletMode::residual) ||
            !std::isfinite(constraint.value)))) {
        return false;
      }
      const auto a = local_to_global.emplace(local, global);
      const auto b = global_to_local.emplace(global, local);
      const int role = row < Topology::velocity_dofs ? row / Topology::velocity_nodes : Dim;
      const auto r = roles.emplace(local, role);
      const auto c = constraints.emplace(local, constraint);
      if ((!r.second && r.first->second != role) ||
          (!c.second && (c.first->second.mode != constraint.mode ||
                         (constraint.mode != DirichletMode::none &&
                          c.first->second.value != constraint.value)))) {
        return false;
      }
      if ((!a.second && a.first->second != global) || (!b.second && b.first->second != local)) {
        return false;
      }
    }
  }
  for (std::size_t row = 0; row < owned.getLocalNumElements(); ++row) {
    if (global_to_local.find(owned.getGlobalElement(row)) == global_to_local.end()) {
      return false;
    }
  }
  return true;
}

template <int Dim> class PreparedFlowState final : public detail::ContextAssemblyState {
public:
  PreparedFlowState(const HostFlowSnapshot<Dim> &snapshot,
                    TpetraSparseMatrix &adapter,
                    GomaSparseMatrix wrapper,
                    double penalty)
      : comm_(adapter.row_map->getComm()), owned_rows_(adapter.row_map->getLocalNumElements()),
        penalty_(penalty) {
    using Topology = Q2Q1<Dim>;
    require_all(*comm_, valid_snapshot(snapshot, *adapter.row_map, penalty),
                "Invalid Kokkos context snapshot, constraints, DOF map, or owned-row coverage");
    require_agreement(*comm_, Dim, "Kokkos context dimensions differ between ranks");
    require_agreement(*comm_, snapshot.expressions.transient ? 1 : 0,
                      "Kokkos context time integration differs between ranks");
    double minimum = 0.0, maximum = 0.0;
    Teuchos::reduceAll(*comm_, Teuchos::REDUCE_MIN, 1, &penalty_, &minimum);
    Teuchos::reduceAll(*comm_, Teuchos::REDUCE_MAX, 1, &penalty_, &maximum);
    if (minimum != maximum) {
      throw std::invalid_argument("Kokkos context Dirichlet penalties differ between ranks");
    }
    host_ = snapshot;
    field_map_ =
        std::make_unique<HostFieldMap<Dim>>(host_.local_dofs, host_.elements, host_.unknowns);
    coordinates_ = ElementCoordinates("goma_context_coordinates", host_.elements,
                                      Topology::velocity_nodes, Dim);
    global_dofs_ = Kokkos::View<GO **>("goma_context_global_dofs", host_.elements, Topology::dofs);
    constraints_ = ElementConstraints("goma_context_constraints", host_.elements, Topology::dofs);
    materials_ = Kokkos::View<ConstantFluid<Dim> *>("goma_context_materials", host_.elements);
    expressions_ = Kokkos::View<FlowExpressions<Dim> *>("goma_context_expressions", host_.elements);
    owned_residual_ = Kokkos::View<double *>("goma_context_owned_residual", owned_rows_);
    auto x = Kokkos::create_mirror_view(coordinates_);
    auto g = Kokkos::create_mirror_view(global_dofs_);
    auto c = Kokkos::create_mirror_view(constraints_);
    auto m = Kokkos::create_mirror_view(materials_);
    auto f = Kokkos::create_mirror_view(expressions_);
    for (std::size_t e = 0; e < host_.elements; ++e) {
      m(e) = host_.material;
      f(e) = host_.expressions;
      for (int node = 0; node < Topology::velocity_nodes; ++node) {
        for (int d = 0; d < Dim; ++d) {
          x(e, node, d) = host_.coordinates[(e * Topology::velocity_nodes + node) * Dim + d];
        }
      }
      for (int row = 0; row < Topology::dofs; ++row) {
        g(e, row) = host_.global_dofs[e * Topology::dofs + row];
        c(e, row) = host_.constraints[e * Topology::dofs + row];
      }
    }
    Kokkos::deep_copy(coordinates_, x);
    Kokkos::deep_copy(global_dofs_, g);
    Kokkos::deep_copy(constraints_, c);
    Kokkos::deep_copy(materials_, m);
    Kokkos::deep_copy(expressions_, f);
    geometry_ = prepare_q2q1_geometry<Dim>(coordinates_);
    const auto status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), geometry_.status);
    bool valid = true;
    for (std::size_t e = 0; e < host_.elements; ++e) {
      valid = valid && status(e) == 0;
    }
    require_all(*comm_, valid, "Kokkos context geometry is invalid on one or more ranks");
    scatter_ = std::make_unique<TpetraScatterPlan>(wrapper, global_dofs_);
  }

  void assemble(const double *solution,
                std::size_t solution_size,
                const double *velocity_dot,
                std::size_t velocity_dot_size,
                double time_coefficient,
                goma_kokkos_evaluation_request request,
                double *owned_residual,
                std::size_t residual_size) override {
    using Topology = Q2Q1<Dim>;
    const bool residual =
        request == GOMA_KOKKOS_RESIDUAL || request == GOMA_KOKKOS_RESIDUAL_AND_JACOBIAN;
    const bool jacobian =
        request == GOMA_KOKKOS_JACOBIAN || request == GOMA_KOKKOS_RESIDUAL_AND_JACOBIAN;
    require_all(*comm_,
                (residual || jacobian) && solution_size == host_.unknowns &&
                    (solution != nullptr || solution_size == 0) &&
                    std::isfinite(time_coefficient) &&
                    (!host_.expressions.transient ||
                     (velocity_dot_size == host_.unknowns &&
                      (velocity_dot != nullptr || velocity_dot_size == 0))) &&
                    (!residual || (residual_size == owned_rows_ &&
                                   (owned_residual != nullptr || residual_size == 0))),
                "Invalid Kokkos context evaluation request or host vector extents");
    require_agreement(*comm_, static_cast<int>(request),
                      "Kokkos evaluation requests differ between ranks");
    bool valid = true;
    for (std::size_t i = 0; i < host_.constraints.size(); ++i) {
      const auto constraint = host_.constraints[i];
      if (constraint.mode == DirichletMode::hard_set) {
        const int index = host_.local_dofs[i];
        valid = valid && solution[index] == constraint.value;
        if (host_.expressions.transient && i % Topology::dofs < Topology::velocity_dofs) {
          valid = valid && velocity_dot[index] == 0.0;
        }
      }
    }
    if (residual) {
      for (std::size_t row = 0; row < residual_size; ++row) {
        valid = valid && std::isfinite(owned_residual[row]);
      }
    }
    require_all(*comm_, valid,
                "Kokkos context requires finite residuals and imposed hard-set fields");
    const auto dot = host_.expressions.transient
                         ? std::span<const double>(velocity_dot, velocity_dot_size)
                         : std::span<const double>();
    fields_ = field_map_->gather({solution, solution_size}, dot, host_.expressions.transient);
    const auto evaluation = request == GOMA_KOKKOS_RESIDUAL   ? EvaluationRequest::residual
                            : request == GOMA_KOKKOS_JACOBIAN ? EvaluationRequest::jacobian
                                                              : EvaluationRequest::both;
    result_ = evaluate_q2q1_elements(geometry_, fields_.solution, fields_.velocity_dot, materials_,
                                     expressions_, time_coefficient, evaluation);
    apply_q2q1_dirichlet(result_, fields_.solution, constraints_, evaluation, penalty_);
    using HostValues =
        Kokkos::View<double *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    if (residual) {
      Kokkos::deep_copy(owned_residual_, HostValues(owned_residual, residual_size));
    }
    scatter_->add(result_.residual, result_.jacobian, result_.status, owned_residual_, evaluation);
    if (residual) {
      Kokkos::deep_copy(HostValues(owned_residual, residual_size), owned_residual_);
    }
  }

private:
  Teuchos::RCP<const Teuchos::Comm<int>> comm_;
  std::size_t owned_rows_;
  double penalty_;
  HostFlowSnapshot<Dim> host_;
  std::unique_ptr<HostFieldMap<Dim>> field_map_;
  ElementCoordinates coordinates_;
  ElementGeometry<Dim> geometry_;
  Kokkos::View<GO **> global_dofs_;
  ElementConstraints constraints_;
  Kokkos::View<ConstantFluid<Dim> *> materials_;
  Kokkos::View<FlowExpressions<Dim> *> expressions_;
  Kokkos::View<double *> owned_residual_;
  GatheredFlowFields<Dim> fields_;
  ElementResult<Dim> result_;
  std::unique_ptr<TpetraScatterPlan> scatter_;
};
} // namespace

template <int Dim>
void setup_context(goma_kokkos_context *context,
                   const HostFlowSnapshot<Dim> &snapshot,
                   GomaSparseMatrix matrix,
                   double dirichlet_penalty) {
  try {
    if (matrix == nullptr || matrix->type != GOMA_SPARSE_MATRIX_TYPE_TPETRA ||
        matrix->data == nullptr) {
      throw std::invalid_argument("A configured Tpetra adapter is required on every rank");
    }
    auto &adapter = *static_cast<TpetraSparseMatrix *>(matrix->data);
    if (adapter.row_map.is_null() || adapter.matrix.is_null()) {
      throw std::invalid_argument("A configured Tpetra matrix and owned row map are required");
    }
    require_all(*adapter.row_map->getComm(),
                context != nullptr && Kokkos::is_initialized() && !Kokkos::is_finalized() &&
                    !context->assembly,
                "Kokkos context must be created, unconfigured, and inside an active runtime");
    auto prepared =
        std::make_unique<PreparedFlowState<Dim>>(snapshot, adapter, matrix, dirichlet_penalty);
    context->assembly = std::move(prepared);
    context->last_error.clear();
  } catch (const std::exception &error) {
    detail::context_error(context, error.what());
    throw;
  } catch (...) {
    detail::context_error(context, "Unknown Kokkos context setup failure");
    throw;
  }
}

template void
setup_context<2>(goma_kokkos_context *, const HostFlowSnapshot<2> &, GomaSparseMatrix, double);
template void
setup_context<3>(goma_kokkos_context *, const HostFlowSnapshot<3> &, GomaSparseMatrix, double);
} // namespace goma::assembly
