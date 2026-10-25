#include <Kokkos_Core.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstring>

#include "mm_kokkos_assembly.h"

namespace {
enum class runtime_mode { owned, borrowed, finalized };
runtime_mode mode = runtime_mode::owned;

struct runtime_session {
  bool active = false;
  ~runtime_session() {
    if (active) {
      goma_kokkos_runtime_finalize();
    }
  }
};

struct assembly_context {
  goma_kokkos_context *value = nullptr;
  ~assembly_context() { goma_kokkos_context_destroy(&value); }
};
} // namespace

TEST_CASE("Kokkos assembly contexts respect runtime ownership", "[assembly][kokkos][runtime]") {
  runtime_session session;
  assembly_context first;
  assembly_context second;

  REQUIRE(goma_kokkos_runtime_check_finalize() == GOMA_KOKKOS_BAD_STATE);
  REQUIRE(goma_kokkos_runtime_finalize() == GOMA_KOKKOS_BAD_STATE);
  REQUIRE(goma_kokkos_context_create(&first.value) == GOMA_KOKKOS_BAD_STATE);
  REQUIRE(first.value == nullptr);
  if (mode == runtime_mode::finalized) {
    REQUIRE(goma_kokkos_runtime_initialize() == GOMA_KOKKOS_BAD_STATE);
    REQUIRE_FALSE(Kokkos::is_initialized());
    REQUIRE(Kokkos::is_finalized());
    return;
  }

  REQUIRE(goma_kokkos_runtime_initialize() == GOMA_KOKKOS_SUCCESS);
  session.active = true;
  REQUIRE(Kokkos::is_initialized());
  REQUIRE(goma_kokkos_runtime_initialize() == GOMA_KOKKOS_BAD_STATE);
  REQUIRE(goma_kokkos_context_create(nullptr) == GOMA_KOKKOS_INVALID_ARGUMENT);
  REQUIRE(goma_kokkos_context_destroy(nullptr) == GOMA_KOKKOS_INVALID_ARGUMENT);
  REQUIRE(goma_kokkos_context_destroy(&first.value) == GOMA_KOKKOS_SUCCESS);

  REQUIRE(goma_kokkos_context_create(&first.value) == GOMA_KOKKOS_SUCCESS);
  REQUIRE(first.value != nullptr);
  REQUIRE(goma_kokkos_context_assemble(first.value, nullptr, 0, nullptr, 0, 0.0,
                                       GOMA_KOKKOS_RESIDUAL, nullptr, 0) == GOMA_KOKKOS_BAD_STATE);
  REQUIRE(std::strlen(goma_kokkos_context_last_error(first.value)) > 0);
  REQUIRE(goma_kokkos_context_assemble(nullptr, nullptr, 0, nullptr, 0, 0.0, GOMA_KOKKOS_RESIDUAL,
                                       nullptr, 0) == GOMA_KOKKOS_INVALID_ARGUMENT);
  auto *original = first.value;
  REQUIRE(goma_kokkos_context_create(&first.value) == GOMA_KOKKOS_INVALID_ARGUMENT);
  REQUIRE(first.value == original);
  REQUIRE(goma_kokkos_context_create(&second.value) == GOMA_KOKKOS_SUCCESS);
  REQUIRE(first.value != second.value);
  REQUIRE(goma_kokkos_runtime_check_finalize() == GOMA_KOKKOS_BAD_STATE);
  REQUIRE(goma_kokkos_runtime_finalize() == GOMA_KOKKOS_BAD_STATE);
  REQUIRE(Kokkos::is_initialized());

  REQUIRE(goma_kokkos_context_destroy(&first.value) == GOMA_KOKKOS_SUCCESS);
  REQUIRE(first.value == nullptr);
  REQUIRE(goma_kokkos_runtime_check_finalize() == GOMA_KOKKOS_BAD_STATE);
  REQUIRE(goma_kokkos_runtime_finalize() == GOMA_KOKKOS_BAD_STATE);
  REQUIRE(goma_kokkos_context_destroy(&second.value) == GOMA_KOKKOS_SUCCESS);
  REQUIRE(second.value == nullptr);
  REQUIRE(Kokkos::is_initialized());

  // Sequential contexts share the runtime, rather than trying to reinitialize Kokkos.
  REQUIRE(goma_kokkos_context_create(&first.value) == GOMA_KOKKOS_SUCCESS);
  REQUIRE(goma_kokkos_context_destroy(&first.value) == GOMA_KOKKOS_SUCCESS);
  REQUIRE(goma_kokkos_runtime_check_finalize() == GOMA_KOKKOS_SUCCESS);
  REQUIRE(goma_kokkos_runtime_finalize() == GOMA_KOKKOS_SUCCESS);
  session.active = false;
  REQUIRE(goma_kokkos_context_create(&first.value) == GOMA_KOKKOS_BAD_STATE);
  REQUIRE(goma_kokkos_runtime_check_finalize() == GOMA_KOKKOS_BAD_STATE);
  REQUIRE(goma_kokkos_runtime_finalize() == GOMA_KOKKOS_BAD_STATE);

  if (mode == runtime_mode::borrowed) {
    REQUIRE(Kokkos::is_initialized());
    REQUIRE_FALSE(Kokkos::is_finalized());
    // A second Goma library session may borrow the same host-owned runtime.
    REQUIRE(goma_kokkos_runtime_initialize() == GOMA_KOKKOS_SUCCESS);
    session.active = true;
    REQUIRE(goma_kokkos_context_create(&first.value) == GOMA_KOKKOS_SUCCESS);
    REQUIRE(goma_kokkos_context_destroy(&first.value) == GOMA_KOKKOS_SUCCESS);
    REQUIRE(goma_kokkos_runtime_check_finalize() == GOMA_KOKKOS_SUCCESS);
    REQUIRE(goma_kokkos_runtime_finalize() == GOMA_KOKKOS_SUCCESS);
    session.active = false;
    REQUIRE(Kokkos::is_initialized());
  } else {
    REQUIRE_FALSE(Kokkos::is_initialized());
    REQUIRE(Kokkos::is_finalized());
    REQUIRE(goma_kokkos_runtime_initialize() == GOMA_KOKKOS_BAD_STATE);
  }
}

int main(int argc, char **argv) {
  if (argc > 1 && std::strcmp(argv[1], "--borrow-runtime") == 0) {
    mode = runtime_mode::borrowed;
  } else if (argc > 1 && std::strcmp(argv[1], "--finalized-runtime") == 0) {
    mode = runtime_mode::finalized;
  }
  if (mode != runtime_mode::owned) {
    for (int i = 1; i < argc; ++i) {
      argv[i] = argv[i + 1];
    }
    --argc;
    Kokkos::initialize();
    if (mode == runtime_mode::finalized) {
      Kokkos::finalize();
    }
  }

  const int result = Catch::Session().run(argc, argv);
  if (Kokkos::is_initialized()) {
    Kokkos::finalize();
  }
  return result;
}
