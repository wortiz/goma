#include "mm_kokkos_assembly.h"

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
} // namespace

struct goma_kokkos_context {
  // Kernels will report device errors here for collective host-side handling.
  Kokkos::View<int> device_status{"goma_assembly_device_status"};
};

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
