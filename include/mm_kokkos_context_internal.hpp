/* Host-only opaque-context state. Runtime-only builds need no solver/Tpetra headers. */
#ifndef GOMA_MM_KOKKOS_CONTEXT_INTERNAL_HPP
#define GOMA_MM_KOKKOS_CONTEXT_INTERNAL_HPP

#include "mm_kokkos_assembly.h"
#include <Kokkos_Core.hpp>
#include <memory>
#include <string>

namespace goma::assembly::detail {
// Dispatch is on the host. Device expression and element kernels remain static.
class ContextAssemblyState {
public:
  virtual ~ContextAssemblyState() = default;
  virtual void assemble(const double *solution,
                        std::size_t solution_size,
                        const double *velocity_dot,
                        std::size_t velocity_dot_size,
                        double time_coefficient,
                        goma_kokkos_evaluation_request request,
                        double *owned_residual,
                        std::size_t residual_size) = 0;
};
} // namespace goma::assembly::detail

struct goma_kokkos_context {
  Kokkos::View<int> device_status{"goma_assembly_device_status"};
  std::string last_error;
  std::unique_ptr<goma::assembly::detail::ContextAssemblyState> assembly;
};

namespace goma::assembly::detail {
inline void context_error(goma_kokkos_context *context, const char *message) noexcept {
  if (context != nullptr) {
    try {
      context->last_error = message;
    } catch (...) {
      context->last_error.clear();
    }
  }
}
} // namespace goma::assembly::detail
#endif
