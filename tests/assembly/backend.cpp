#include <catch2/catch_test_macros.hpp>
#include <string>

#include "mm_assembly.h"

TEST_CASE("Assembly backend selection is explicit", "[assembly]") {
  goma_assembly_backend backend = GOMA_ASSEMBLY_KOKKOS;
  REQUIRE(goma_assembly_backend_parse("legacy", &backend) == 0);
  REQUIRE(backend == GOMA_ASSEMBLY_LEGACY);
  REQUIRE(goma_assembly_backend_parse("kokkos", &backend) == 0);
  REQUIRE(backend == GOMA_ASSEMBLY_KOKKOS);
  REQUIRE(goma_assembly_backend_parse("Kokkos", &backend) != 0);
  REQUIRE(backend == GOMA_ASSEMBLY_KOKKOS);
  REQUIRE(goma_assembly_backend_parse("", &backend) != 0);
  REQUIRE(goma_assembly_backend_parse(nullptr, &backend) != 0);
  REQUIRE(goma_assembly_backend_parse("legacy", nullptr) != 0);
}

TEST_CASE("Assembly build availability is checked before complete-system preflight", "[assembly]") {
  REQUIRE(goma_assembly_backend_check(GOMA_ASSEMBLY_LEGACY, 0) == nullptr);
  REQUIRE(goma_assembly_backend_check(GOMA_ASSEMBLY_LEGACY, 1) == nullptr);
  REQUIRE(goma_assembly_backend_check(static_cast<goma_assembly_backend>(-1), 0) != nullptr);
#ifdef GOMA_ENABLE_KOKKOS_ASSEMBLY
  // Build selection succeeds; setup must inspect the complete coupled problem.
  REQUIRE(goma_assembly_backend_check(GOMA_ASSEMBLY_KOKKOS, 0) == nullptr);
  REQUIRE(goma_assembly_backend_check(GOMA_ASSEMBLY_KOKKOS, 1) == nullptr);
#else
  const std::string error = goma_assembly_backend_check(GOMA_ASSEMBLY_KOKKOS, 0);
  const std::string ad_error = goma_assembly_backend_check(GOMA_ASSEMBLY_KOKKOS, 1);
  REQUIRE(error.find("ENABLE_KOKKOS_ASSEMBLY=ON") != std::string::npos);
  REQUIRE(ad_error == error);
#endif
}
