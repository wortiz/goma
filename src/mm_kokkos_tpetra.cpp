#include "mm_kokkos_tpetra.hpp"

#include "linalg/sparse_matrix_tpetra.h"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace goma::assembly {
namespace {
void collective_require(const Teuchos::Comm<int> &comm, bool valid, const char *message) {
  const int local_error = valid ? 0 : 1;
  int global_error = 0;
  Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &local_error, &global_error);
  if (global_error != 0) {
    throw std::invalid_argument(message);
  }
}
} // namespace

struct TpetraScatterPlan::Impl {
  using Matrix = Tpetra::FECrsMatrix<double, LO, GO>;
  static_assert(Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace,
                                           Matrix::device_type::memory_space>::accessible,
                "Kokkos assembly must be able to access Tpetra's device memory");
  TpetraSparseMatrix *adapter;
  Teuchos::RCP<Matrix> matrix;
  Teuchos::RCP<const Matrix::crs_graph_type> graph;
  Teuchos::RCP<const Matrix::map_type> row_map;
  Teuchos::RCP<const Matrix::map_type> col_map;
  Teuchos::RCP<const Teuchos::Comm<int>> comm;
  Kokkos::View<LO **> rows;
  Kokkos::View<std::size_t ***> offsets;
  std::size_t elements;
  std::size_t dofs;
};

TpetraScatterPlan::TpetraScatterPlan(GomaSparseMatrix wrapper,
                                     const Kokkos::View<const GomaGlobalOrdinal **> &element_dofs) {
  if (wrapper == nullptr || wrapper->type != GOMA_SPARSE_MATRIX_TYPE_TPETRA ||
      wrapper->data == nullptr) {
    throw std::invalid_argument("A valid Tpetra adapter is required on every rank");
  }
  auto *adapter = static_cast<TpetraSparseMatrix *>(wrapper->data);
  if (adapter->matrix.is_null() || adapter->row_map.is_null()) {
    throw std::invalid_argument("A configured Tpetra matrix and owned row map are required");
  }
  impl_ = std::make_unique<Impl>();
  auto &plan = *impl_;
  plan.adapter = adapter;
  plan.matrix = adapter->matrix;
  plan.graph = plan.matrix->getCrsGraph();
  plan.row_map = plan.matrix->getRowMap();
  plan.col_map = plan.matrix->getColMap();
  plan.comm = adapter->row_map->getComm();
  plan.elements = element_dofs.extent(0);
  plan.dofs = element_dofs.extent(1);
  collective_require(*plan.comm,
                     !plan.graph.is_null() && !plan.row_map.is_null() && !plan.col_map.is_null() &&
                         plan.matrix->isStaticGraph() && plan.graph->isFillComplete() &&
                         plan.graph->isLocallyIndexed() && plan.elements <= INT_MAX &&
                         plan.dofs > 0 && plan.dofs <= INT_MAX,
                     "Tpetra scatter requires a local static graph and valid element extents");
  collective_require(*plan.comm, plan.row_map->isSameAs(*adapter->row_map),
                     "Tpetra scatter requires an owned-only active row map");
  plan.rows = Kokkos::View<LO **>("goma_tpetra_owned_rows", plan.elements, plan.dofs);
  plan.offsets =
      Kokkos::View<std::size_t ***>("goma_tpetra_offsets", plan.elements, plan.dofs, plan.dofs);
  auto rows = Kokkos::create_mirror_view(plan.rows);
  auto offsets = Kokkos::create_mirror_view(plan.offsets);
  const auto gids = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), element_dofs);
  const auto graph = plan.graph->getLocalGraphHost();
  const bool sorted = plan.graph->isSorted();
  bool valid = true;
  for (std::size_t e = 0; e < plan.elements; ++e) {
    std::unordered_set<GO> unique;
    for (std::size_t row = 0; row < plan.dofs; ++row) {
      valid = unique.insert(gids(e, row)).second && valid;
      const LO local_row = plan.row_map->getLocalElement(gids(e, row));
      rows(e, row) = local_row;
      for (std::size_t column = 0; column < plan.dofs; ++column) {
        offsets(e, row, column) = std::numeric_limits<std::size_t>::max();
        if (local_row == Teuchos::OrdinalTraits<LO>::invalid()) {
          continue;
        }
        const LO local_column = plan.col_map->getLocalElement(gids(e, column));
        if (local_column == Teuchos::OrdinalTraits<LO>::invalid()) {
          valid = false;
          continue;
        }
        const auto begin = graph.entries.data() + graph.row_map(local_row);
        const auto end = graph.entries.data() + graph.row_map(local_row + 1);
        const auto found = sorted ? std::lower_bound(begin, end, local_column)
                                  : std::find(begin, end, local_column);
        if (found == end || *found != local_column) {
          valid = false;
        } else {
          offsets(e, row, column) = static_cast<std::size_t>(found - graph.entries.data());
        }
      }
    }
  }
  collective_require(*plan.comm, valid,
                     "Tpetra scatter has duplicate element DOFs or missing graph entries");
  Kokkos::deep_copy(plan.rows, rows);
  Kokkos::deep_copy(plan.offsets, offsets);
}

TpetraScatterPlan::~TpetraScatterPlan() = default;

void TpetraScatterPlan::add(const Kokkos::View<const double **> &element_residual,
                            const Kokkos::View<const double ***> &element_jacobian,
                            const Kokkos::View<const int *> &element_status,
                            const Kokkos::View<double *> &owned_residual,
                            EvaluationRequest request) const {
  const auto &plan = *impl_;
  const bool residual =
      request == EvaluationRequest::residual || request == EvaluationRequest::both;
  const bool jacobian =
      request == EvaluationRequest::jacobian || request == EvaluationRequest::both;
  const bool same_matrix = plan.adapter->matrix.getRawPtr() == plan.matrix.getRawPtr();
  collective_require(
      *plan.comm,
      (residual || jacobian) && same_matrix && (!jacobian || plan.matrix->isFillActive()) &&
          plan.matrix->getCrsGraph().getRawPtr() == plan.graph.getRawPtr() &&
          plan.matrix->getRowMap().getRawPtr() == plan.row_map.getRawPtr() &&
          plan.matrix->getColMap().getRawPtr() == plan.col_map.getRawPtr() &&
          plan.graph->isFillComplete() && element_status.extent(0) == plan.elements &&
          (!residual || (element_residual.extent(0) == plan.elements &&
                         element_residual.extent(1) == plan.dofs &&
                         owned_residual.extent(0) == plan.row_map->getLocalNumElements())) &&
          (!jacobian ||
           (element_jacobian.extent(0) == plan.elements &&
            element_jacobian.extent(1) == plan.dofs && element_jacobian.extent(2) == plan.dofs)),
      "Invalid Tpetra scatter inputs, matrix phase, or stale graph/map cache");
  const auto status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), element_status);
  bool valid = true;
  for (std::size_t e = 0; e < plan.elements; ++e) {
    valid = valid && status(e) == 0;
  }
  collective_require(*plan.comm, valid,
                     "Element evaluation failed; global scatter was not started");
  const auto rows = plan.rows;
  const auto offsets = plan.offsets;
  const int dofs = static_cast<int>(plan.dofs);
  const LO invalid_row = Teuchos::OrdinalTraits<LO>::invalid();
  using Policy = Kokkos::TeamPolicy<>;
  using Team = typename Policy::member_type;
  const Policy policy(static_cast<int>(plan.elements), Kokkos::AUTO);
  if (jacobian) {
    // Acquire writable device values only here, and release after the fence.
    // This is Tpetra's synchronization contract; no host row insertion is used.
    const auto values = plan.matrix->getLocalValuesDevice(Tpetra::Access::ReadWrite);
    Kokkos::parallel_for(
        "goma_tpetra_element_scatter", policy, KOKKOS_LAMBDA(const Team &team) {
          const int element = team.league_rank();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, dofs), [&](int row) {
            const LO owned = rows(element, row);
            if (owned == invalid_row) {
              return;
            }
            if (residual) {
              Kokkos::atomic_add(&owned_residual(owned), element_residual(element, row));
            }
            for (int column = 0; column < dofs; ++column) {
              Kokkos::atomic_add(&values(offsets(element, row, column)),
                                 element_jacobian(element, row, column));
            }
          });
        });
    Kokkos::fence("Goma Tpetra matrix scatter");
  } else {
    Kokkos::parallel_for(
        "goma_tpetra_residual_scatter", policy, KOKKOS_LAMBDA(const Team &team) {
          const int element = team.league_rank();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, dofs), [&](int row) {
            const LO owned = rows(element, row);
            if (owned != invalid_row) {
              Kokkos::atomic_add(&owned_residual(owned), element_residual(element, row));
            }
          });
        });
    Kokkos::fence("Goma Tpetra residual scatter");
  }
}

} // namespace goma::assembly
