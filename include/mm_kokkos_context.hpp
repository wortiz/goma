/* One-time setup of the opaque context from an owned host snapshot. */
#ifndef GOMA_MM_KOKKOS_CONTEXT_HPP
#define GOMA_MM_KOKKOS_CONTEXT_HPP

#include "mm_kokkos_assembly.h"
#include "mm_kokkos_host.hpp"

namespace goma::assembly {
// Collective on a configured Tpetra adapter's communicator. Copy all snapshot
// data; the caller may release or modify it after return. The adapter/static
// graph and runtime must outlive the context. Failed setup leaves it unconfigured
// and retryable; successful setup may not be replaced. Recreate the context for
// a new mesh, map, graph, or material configuration. Throws on invalid setup.
template <int Dim>
void setup_context(goma_kokkos_context *context,
                   const HostFlowSnapshot<Dim> &snapshot,
                   GomaSparseMatrix matrix,
                   double dirichlet_penalty);
} // namespace goma::assembly
#endif
