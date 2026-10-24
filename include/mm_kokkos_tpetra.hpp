/* Owned-row device scatter into the existing Tpetra adapter. */
#ifndef GOMA_MM_KOKKOS_TPETRA_HPP
#define GOMA_MM_KOKKOS_TPETRA_HPP

#include "linalg/sparse_matrix.h"
#include "mm_kokkos_element.hpp"
#include <memory>

namespace goma::assembly {

// Collective on the matrix communicator. The adapter and its static graph must
// remain alive and unchanged for the plan's lifetime. Only owned rows are written;
// the caller must provide all incident elements, including ghost elements, once.
// Rebuild the plan after any graph/map change. Keep it inside the Kokkos lifetime.
class TpetraScatterPlan {
public:
  TpetraScatterPlan(GomaSparseMatrix matrix,
                    const Kokkos::View<const GomaGlobalOrdinal **> &element_dofs);
  ~TpetraScatterPlan();
  TpetraScatterPlan(const TpetraScatterPlan &) = delete;
  TpetraScatterPlan &operator=(const TpetraScatterPlan &) = delete;

  // Add to existing values; does not zero, scale, close, or change FE assembly
  // phases. Jacobian requests require beginAssembly/beginModify before calling
  // and endAssembly/endModify afterwards. Residual-only calls may use a closed matrix. Local errors
  // are rejected collectively before any global writes. The residual follows the owned row-map
  // ordering. Return only after a fence, so Tpetra can safely acquire host values or change
  // assembly phase immediately.
  void add(const Kokkos::View<const double **> &element_residual,
           const Kokkos::View<const double ***> &element_jacobian,
           const Kokkos::View<const int *> &element_status,
           const Kokkos::View<double *> &owned_residual,
           EvaluationRequest request) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace goma::assembly
#endif
