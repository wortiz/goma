#include "mm_kokkos_assembly.h"
#include "mm_kokkos_context_internal.hpp"

#include <stdexcept>

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <memory>

namespace {
bool runtime_active = false;
bool runtime_owned = false;
std::size_t live_contexts = 0;

bool runtime_available() {
  return runtime_active && Kokkos::is_initialized() && !Kokkos::is_finalized();
}
bool finish_failed_assembly(goma_kokkos_context *context) noexcept {
  try {
    Kokkos::fence("Goma context failed assembly completion");
    return true;
  } catch (...) {
    goma::assembly::detail::context_error(context, "Kokkos context could not complete device work");
    return false;
  }
}
} // namespace

extern "C" goma_kokkos_status goma_kokkos_runtime_initialize(void) {
  if (runtime_active || Kokkos::is_finalized()) {
    return GOMA_KOKKOS_BAD_STATE;
  }
  try {
    const bool already_initialized = Kokkos::is_initialized();
    if (!already_initialized) {
      Kokkos::initialize();
    }
    runtime_owned = !already_initialized;
    runtime_active = true;
    return GOMA_KOKKOS_SUCCESS;
  } catch (...) {
    // Exceptions must not cross the C interface.
    return GOMA_KOKKOS_RUNTIME_ERROR;
  }
}

extern "C" goma_kokkos_status goma_kokkos_runtime_check_finalize(void) {
  return runtime_available() && live_contexts == 0 ? GOMA_KOKKOS_SUCCESS : GOMA_KOKKOS_BAD_STATE;
}

extern "C" goma_kokkos_status goma_kokkos_runtime_finalize(void) {
  if (goma_kokkos_runtime_check_finalize() != GOMA_KOKKOS_SUCCESS) {
    return GOMA_KOKKOS_BAD_STATE;
  }
  try {
    if (runtime_owned) {
      Kokkos::finalize();
    }
    runtime_owned = false;
    runtime_active = false;
    return GOMA_KOKKOS_SUCCESS;
  } catch (...) {
    return GOMA_KOKKOS_RUNTIME_ERROR;
  }
}

extern "C" goma_kokkos_status goma_kokkos_context_create(goma_kokkos_context **context) {
  if (context == nullptr || *context != nullptr) {
    return GOMA_KOKKOS_INVALID_ARGUMENT;
  }
  if (!runtime_available()) {
    return GOMA_KOKKOS_BAD_STATE;
  }
  try {
    auto created = std::make_unique<goma_kokkos_context>();
    *context = created.release();
    ++live_contexts;
    return GOMA_KOKKOS_SUCCESS;
  } catch (...) {
    return GOMA_KOKKOS_RUNTIME_ERROR;
  }
}

extern "C" goma_kokkos_status goma_kokkos_context_assemble(goma_kokkos_context *context,
                                                           const double *solution,
                                                           size_t solution_size,
                                                           const double *velocity_dot,
                                                           size_t velocity_dot_size,
                                                           double time_coefficient,
                                                           goma_kokkos_evaluation_request request,
                                                           double *owned_residual,
                                                           size_t residual_size) {
  using goma::assembly::detail::context_error;
  if (context == nullptr) {
    return GOMA_KOKKOS_INVALID_ARGUMENT;
  }
  if (!runtime_available() || !context->assembly) {
    context_error(context, "Kokkos context requires an active runtime and completed setup");
    return GOMA_KOKKOS_BAD_STATE;
  }
  try {
    context->assembly->assemble(solution, solution_size, velocity_dot, velocity_dot_size,
                                time_coefficient, request, owned_residual, residual_size);
    context->last_error.clear();
    return GOMA_KOKKOS_SUCCESS;
  } catch (const std::invalid_argument &error) {
    context_error(context, error.what());
    return finish_failed_assembly(context) ? GOMA_KOKKOS_INVALID_ARGUMENT
                                           : GOMA_KOKKOS_RUNTIME_ERROR;
  } catch (const std::exception &error) {
    context_error(context, error.what());
    finish_failed_assembly(context);
    return GOMA_KOKKOS_RUNTIME_ERROR;
  } catch (...) {
    context_error(context, "Unknown Kokkos context evaluation failure");
    finish_failed_assembly(context);
    return GOMA_KOKKOS_RUNTIME_ERROR;
  }
}

extern "C" const char *goma_kokkos_context_last_error(const goma_kokkos_context *context) {
  return context == nullptr ? "No Kokkos assembly context" : context->last_error.c_str();
}

extern "C" goma_kokkos_status goma_kokkos_context_destroy(goma_kokkos_context **context) {
  if (context == nullptr) {
    return GOMA_KOKKOS_INVALID_ARGUMENT;
  }
  if (*context == nullptr) {
    return GOMA_KOKKOS_SUCCESS;
  }
  if (!runtime_available()) {
    return GOMA_KOKKOS_BAD_STATE;
  }
  try {
    Kokkos::fence("Goma assembly context destruction");
    delete *context;
    *context = nullptr;
    --live_contexts;
    return GOMA_KOKKOS_SUCCESS;
  } catch (...) {
    return GOMA_KOKKOS_RUNTIME_ERROR;
  }
}
