#include <Kokkos_Core.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

#include "mm_kokkos_flow.hpp"

namespace {
using namespace goma::assembly;

void check_close(double actual, double expected, double tolerance = 1.e-9) {
  INFO("actual = " << actual << ", expected = " << expected);
  CHECK(std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected)));
}

template <int Dim> FlowState<double, Dim> sample_state() {
  FlowState<double, Dim> state;
  state.pressure = 2.3;
  for (int d = 0; d < Dim; ++d) {
    state.velocity[d] = (d % 2 == 0 ? 1.0 : -1.0) * (0.8 + 0.7 * d);
    state.velocity_dot[d] = -0.2 + 0.3 * d;
    for (int a = 0; a < Dim; ++a) {
      state.gradient[d][a] = 0.25 + 0.4 * d - 0.3 * a + 0.15 * d * a;
    }
  }
  return state;
}

template <int Dim> ConstantFluid<Dim> sample_fluid() {
  ConstantFluid<Dim> fluid;
  fluid.density = 1.9;
  fluid.viscosity = 0.37;
  for (int d = 0; d < Dim; ++d) {
    fluid.source[d] = 0.2 - 1.3 * d;
  }
  return fluid;
}

template <int Dim> Basis<Dim> test_basis() {
  Basis<Dim> basis;
  basis.value = 0.63;
  for (int d = 0; d < Dim; ++d) {
    basis.gradient[d] = 0.22 - 0.31 * d;
  }
  return basis;
}

template <int Dim> Basis<Dim> trial_basis() {
  Basis<Dim> basis;
  basis.value = -0.41;
  for (int d = 0; d < Dim; ++d) {
    basis.gradient[d] = 0.3 + 0.23 * d;
  }
  return basis;
}

template <int Dim>
FlowState<double, Dim>
shifted(FlowState<double, Dim> state, const FlowState<double, Dim> &increment, double step) {
  state.pressure += step * increment.pressure;
  for (int d = 0; d < Dim; ++d) {
    state.velocity[d] += step * increment.velocity[d];
    state.velocity_dot[d] += step * increment.velocity_dot[d];
    for (int a = 0; a < Dim; ++a) {
      state.gradient[d][a] += step * increment.gradient[d][a];
    }
  }
  return state;
}

template <int Dim>
MaterialValues<double, Dim> shifted(MaterialValues<double, Dim> material,
                                    const MaterialValues<double, Dim> &tangent,
                                    double step) {
  material.density += step * tangent.density;
  material.viscosity += step * tangent.viscosity;
  for (int d = 0; d < Dim; ++d) {
    material.source[d] += step * tangent.source[d];
  }
  return material;
}

template <int Dim> FlowExpressions<Dim> sample_expressions(bool transient) {
  FlowExpressions<Dim> flow;
  flow.transient = transient;
  for (int a = 0; a < Dim; ++a) {
    flow.momentum[a].mass = {true, 0.8 + 0.2 * a};
    flow.momentum[a].advection = {true, -0.35 - 0.1 * a};
    flow.momentum[a].stress = {true, 1.1 + 0.3 * a};
    flow.momentum[a].source = {true, 0.4 - 0.15 * a};
  }
  flow.continuity = {true, -0.7};
  return flow;
}

template <int Dim, typename Expression> void check_term_derivatives(Expression expression) {
  const auto state = sample_state<Dim>();
  const auto material = sample_fluid<Dim>().evaluate(state);
  const auto test = test_basis<Dim>();
  const auto trial = trial_basis<Dim>();
  constexpr double step = 1.e-5;

  // Also test material tangents separately from the constant-fluid zero tangents.
  for (bool use_material_tangent : {false, true}) {
    MaterialValues<double, Dim> tangent;
    if (use_material_tangent) {
      tangent.density = -0.7;
      tangent.viscosity = 0.21;
      for (int d = 0; d < Dim; ++d) {
        tangent.source[d] = 0.1 + 0.2 * d;
      }
    }
    for (double time_coefficient : {0.0, 1.7}) {
      for (int column = 0; column <= Dim; ++column) {
        const auto increment = column == Dim ? pressure_increment(trial)
                                             : velocity_increment(trial, column, time_coefficient);
        const auto plus = shifted(state, increment, step);
        const auto minus = shifted(state, increment, -step);
        const auto material_plus = shifted(material, tangent, step);
        const auto material_minus = shifted(material, tangent, -step);
        for (int component = 0; component < Dim; ++component) {
          INFO("dimension = " << Dim << ", component = " << component << ", column = " << column
                              << ", time coefficient = " << time_coefficient
                              << ", material tangent = " << use_material_tangent);
          const double difference = (expression.residual(plus, material_plus, test, component) -
                                     expression.residual(minus, material_minus, test, component)) /
                                    (2.0 * step);
          check_close(expression.linearize(state, increment, material, tangent, test, component),
                      difference);
        }
      }
    }
  }
}

template <int Dim> void check_device_expressions(bool transient) {
  const auto state = sample_state<Dim>();
  const auto fluid = sample_fluid<Dim>();
  const auto test = test_basis<Dim>();
  const auto trial = trial_basis<Dim>();
  constexpr double weight = 1.73;
  constexpr double time_coefficient = 2.3;
  constexpr double step = 1.e-5;

  for (bool all_terms : {false, true}) {
    auto flow = sample_expressions<Dim>(transient);
    if (!all_terms) {
      flow.momentum[0].mass.enabled = false;
      flow.momentum[0].advection.multiplier = 0.0;
      flow.momentum[1].stress.enabled = false;
      flow.momentum[1].source.multiplier = 0.0;
      flow.continuity.enabled = false;
    }
    // Each row stores its residual followed by Dim velocity and one pressure derivative.
    Kokkos::View<double **> result("flow expressions", Dim + 1, Dim + 2);
    Kokkos::parallel_for(
        "goma_flow_expression_rows", Dim + 1, KOKKOS_LAMBDA(int row) {
          const auto material = fluid.evaluate(state);
          result(row, 0) = row == Dim ? flow.continuity_residual(state, test, weight)
                                      : flow.momentum_residual(state, material, test, row, weight);
          for (int column = 0; column <= Dim; ++column) {
            const auto increment = column == Dim
                                       ? pressure_increment(trial)
                                       : velocity_increment(trial, column, time_coefficient);
            const auto tangent = fluid.linearize(state, increment);
            result(row, column + 1) = row == Dim
                                          ? flow.continuity_linearization(increment, test, weight)
                                          : flow.momentum_linearization(state, increment, material,
                                                                        tangent, test, row, weight);
          }
        });
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
    const auto material = fluid.evaluate(state);
    for (int row = 0; row <= Dim; ++row) {
      INFO("dimension = " << Dim << ", row = " << row << ", transient = " << transient
                          << ", all terms = " << all_terms);
      const double expected = row == Dim
                                  ? flow.continuity_residual(state, test, weight)
                                  : flow.momentum_residual(state, material, test, row, weight);
      check_close(host(row, 0), expected, 1.e-13);
      for (int column = 0; column <= Dim; ++column) {
        INFO("column = " << column);
        const auto increment = column == Dim ? pressure_increment(trial)
                                             : velocity_increment(trial, column, time_coefficient);
        const auto plus = shifted(state, increment, step);
        const auto minus = shifted(state, increment, -step);
        const double r_plus =
            row == Dim ? flow.continuity_residual(plus, test, weight)
                       : flow.momentum_residual(plus, fluid.evaluate(plus), test, row, weight);
        const double r_minus =
            row == Dim ? flow.continuity_residual(minus, test, weight)
                       : flow.momentum_residual(minus, fluid.evaluate(minus), test, row, weight);
        check_close(host(row, column + 1), (r_plus - r_minus) / (2.0 * step));
      }
    }
  }
}

TEST_CASE("Constant flow material has zero solution tangents", "[assembly][kokkos][flow]") {
  const auto fluid = sample_fluid<3>();
  const auto state = sample_state<3>();
  const auto increment = velocity_increment(trial_basis<3>(), 2, 1.9);
  const auto material = fluid.evaluate(state);
  const auto tangent = fluid.linearize(state, increment);
  CHECK(material.density == fluid.density);
  CHECK(material.viscosity == fluid.viscosity);
  CHECK(tangent.density == 0.0);
  CHECK(tangent.viscosity == 0.0);
  for (int d = 0; d < 3; ++d) {
    CHECK(material.source[d] == fluid.source[d]);
    CHECK(tangent.source[d] == 0.0);
  }
  const auto perturbed = fluid.evaluate(shifted(state, increment, 0.7));
  CHECK(perturbed.density == material.density);
  CHECK(perturbed.viscosity == material.viscosity);
  for (int d = 0; d < 3; ++d) {
    CHECK(perturbed.source[d] == material.source[d]);
  }
}

TEST_CASE("Cartesian flow terms preserve legacy signs and stress convention",
          "[assembly][kokkos][flow]") {
  const FlowState<double, 2> state{{2.0, -1.0}, {{1.0, 2.0}, {-3.0, 4.0}}, {0.5, -0.75}, 3.0};
  const MaterialValues<double, 2> material{2.5, 1.2, {4.0, -2.0}};
  const Basis<2> test{0.4, {0.3, -0.2}};
  check_close(MomentumMass<2>{}.residual(state, material, test, 0), -0.5, 1.e-13);
  check_close(MomentumMass<2>{}.residual(state, material, test, 1), 0.75, 1.e-13);
  check_close(MomentumAdvection<2>{}.residual(state, material, test, 0), -5.0, 1.e-13);
  check_close(MomentumAdvection<2>{}.residual(state, material, test, 1), 0.0, 1.e-13);
  check_close(MomentumStress<2>{}.residual(state, material, test, 0), -0.06, 1.e-13);
  check_close(MomentumStress<2>{}.residual(state, material, test, 1), 1.68, 1.e-13);
  check_close(MomentumSource<2>{}.residual(state, material, test, 0), 1.6, 1.e-13);
  check_close(MomentumSource<2>{}.residual(state, material, test, 1), -0.8, 1.e-13);
  check_close(Continuity<2>{}.residual(state, test), 2.0, 1.e-13);

  const auto trial = trial_basis<2>();
  const auto pressure = pressure_increment(trial);
  const MaterialValues<double, 2> tangent;
  for (int a = 0; a < 2; ++a) {
    check_close(MomentumStress<2>{}.linearize(state, pressure, material, tangent, test, a),
                test.gradient[a] * trial.value, 1.e-13);
    check_close(Continuity<2>{}.linearize(velocity_increment(trial, a, 1.7), test),
                test.value * trial.gradient[a], 1.e-13);
  }
  CHECK(Continuity<2>{}.linearize(pressure, test) == 0.0);

  FlowState<double, 2> rotation;
  rotation.gradient[0][1] = 2.0;
  rotation.gradient[1][0] = -2.0;
  CHECK(MomentumStress<2>{}.residual(rotation, material, test, 0) == 0.0);
  CHECK(MomentumStress<2>{}.residual(rotation, material, test, 1) == 0.0);
}

TEST_CASE("Three-dimensional flow includes third-direction coupling", "[assembly][kokkos][flow]") {
  const FlowState<double, 3> state{{1.0, -2.0, 3.0},
                                   {{1.0, 2.0, -1.0}, {-3.0, 4.0, 0.5}, {2.0, -0.5, -2.0}},
                                   {0.2, -0.4, 0.6},
                                   2.0};
  const MaterialValues<double, 3> material{2.0, 0.5, {1.0, -2.0, 4.0}};
  const Basis<3> test{0.25, {0.2, -0.3, 0.4}};
  const double mass[] = {-0.1, 0.2, -0.3};
  const double advection[] = {-6.5, 3.75, 4.0};
  const double stress[] = {-0.15, 0.7, 1.5};
  const double source[] = {0.25, -0.5, 1.0};
  for (int a = 0; a < 3; ++a) {
    check_close(MomentumMass<3>{}.residual(state, material, test, a), mass[a], 1.e-13);
    check_close(MomentumAdvection<3>{}.residual(state, material, test, a), advection[a], 1.e-13);
    check_close(MomentumStress<3>{}.residual(state, material, test, a), stress[a], 1.e-13);
    check_close(MomentumSource<3>{}.residual(state, material, test, a), source[a], 1.e-13);
  }
  check_close(Continuity<3>{}.residual(state, test), 0.75, 1.e-13);
}

TEST_CASE("Flow composition applies equation multipliers and integration weights",
          "[assembly][kokkos][flow]") {
  const FlowState<double, 2> state{{2.0, -1.0}, {{1.0, 2.0}, {-3.0, 4.0}}, {0.5, -0.75}, 3.0};
  const MaterialValues<double, 2> material{2.5, 1.2, {4.0, -2.0}};
  const Basis<2> test{0.4, {0.3, -0.2}};
  FlowExpressions<2> flow;
  flow.transient = true;
  flow.momentum[0] = {{true, 2.0}, {true, 0.5}, {true, 1.5}, {true, -0.25}};
  flow.momentum[1] = {{true, -1.0}, {true, 2.0}, {true, 0.5}, {true, 3.0}};
  flow.continuity = {true, -0.4};
  check_close(flow.momentum_residual(state, material, test, 0, 1.7), -6.783, 1.e-13);
  check_close(flow.momentum_residual(state, material, test, 1, 1.7), -3.927, 1.e-13);
  check_close(flow.continuity_residual(state, test, 1.7), -1.36, 1.e-13);
  flow.transient = false;
  check_close(flow.momentum_residual(state, material, test, 0, 1.7), -5.083, 1.e-13);
  check_close(flow.momentum_residual(state, material, test, 1, 1.7), -2.652, 1.e-13);
  flow.momentum[0].source.enabled = false;
  flow.momentum[1].stress.multiplier = 0.0;
  check_close(flow.momentum_residual(state, material, test, 0, 1.7), -4.403, 1.e-13);
  check_close(flow.momentum_residual(state, material, test, 1, 1.7), -4.08, 1.e-13);
}

TEST_CASE("Velocity linearization uses the supplied time derivative coefficient",
          "[assembly][kokkos][flow][jacobian]") {
  const auto state = sample_state<3>();
  const auto material = sample_fluid<3>().evaluate(state);
  const MaterialValues<double, 3> tangent;
  const auto trial = trial_basis<3>();
  const auto test = test_basis<3>();
  for (double coefficient : {0.0, 0.7, 4.3}) {
    for (int column = 0; column < 3; ++column) {
      const auto increment = velocity_increment(trial, column, coefficient);
      for (int row = 0; row < 3; ++row) {
        const double expected =
            row == column ? -test.value * material.density * trial.value * coefficient : 0.0;
        check_close(MomentumMass<3>{}.linearize(state, increment, material, tangent, test, row),
                    expected, 1.e-13);
      }
    }
  }
}

TEST_CASE("Flow residuals retain the requested scalar type", "[assembly][kokkos][flow]") {
  FlowState<float, 2> state;
  state.pressure = 1.5f;
  state.velocity[0] = 2.0f;
  state.gradient[0][0] = 0.5f;
  const auto material = sample_fluid<2>().evaluate(state);
  const auto test = test_basis<2>();
  const auto flow = sample_expressions<2>(false);
  static_assert(
      std::is_same_v<decltype(flow.momentum_residual(state, material, test, 0, 1.0)), float>);
  static_assert(std::is_same_v<decltype(flow.continuity_residual(state, test, 1.0)), float>);
  CHECK(std::isfinite(flow.momentum_residual(state, material, test, 0, 1.0)));
}

TEST_CASE("Disabled flow terms do not evaluate unused fields", "[assembly][kokkos][flow]") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  auto state = sample_state<3>();
  for (int d = 0; d < 3; ++d) {
    state.velocity[d] = nan;
    state.velocity_dot[d] = nan;
    for (int a = 0; a < 3; ++a) {
      state.gradient[d][a] = nan;
    }
  }
  state.pressure = nan;
  const MaterialValues<double, 3> material{nan, nan, {nan, nan, nan}};
  const auto test = test_basis<3>();
  FlowExpressions<3> flow;
  for (int component = 0; component < 3; ++component) {
    CHECK(flow.momentum_residual(state, material, test, component, 2.0) == 0.0);
    CHECK(flow.momentum_linearization(state, state, material, material, test, component, 2.0) ==
          0.0);
    // Steady assembly ignores mass even when its term is enabled.
    flow.momentum[component].mass = {true, 2.7};
    CHECK(flow.momentum_residual(state, material, test, component, 2.0) == 0.0);
    CHECK(flow.momentum_linearization(state, state, material, material, test, component, 2.0) ==
          0.0);
    flow.momentum[component] = {{true, 0.0}, {true, 0.0}, {true, 0.0}, {true, 0.0}};
  }
  flow.transient = true;
  for (int component = 0; component < 3; ++component) {
    CHECK(flow.momentum_residual(state, material, test, component, 2.0) == 0.0);
    CHECK(flow.momentum_linearization(state, state, material, material, test, component, 2.0) ==
          0.0);
  }
  CHECK(flow.continuity_residual(state, test, 2.0) == 0.0);
  CHECK(flow.continuity_linearization(state, test, 2.0) == 0.0);
  flow.continuity = {true, 0.0};
  CHECK(flow.continuity_residual(state, test, 2.0) == 0.0);
  CHECK(flow.continuity_linearization(state, test, 2.0) == 0.0);
}

TEST_CASE("Flow term explicit Jacobians match centered differences in 2D",
          "[assembly][kokkos][flow][jacobian]") {
  check_term_derivatives<2>(MomentumMass<2>{});
  check_term_derivatives<2>(MomentumAdvection<2>{});
  check_term_derivatives<2>(MomentumStress<2>{});
  check_term_derivatives<2>(MomentumSource<2>{});
}

TEST_CASE("Flow term explicit Jacobians match centered differences in 3D",
          "[assembly][kokkos][flow][jacobian]") {
  check_term_derivatives<3>(MomentumMass<3>{});
  check_term_derivatives<3>(MomentumAdvection<3>{});
  check_term_derivatives<3>(MomentumStress<3>{});
  check_term_derivatives<3>(MomentumSource<3>{});
}

TEST_CASE("Kokkos evaluates 2D flow residuals and Jacobians", "[assembly][kokkos][flow][device]") {
  const bool transient = GENERATE(false, true);
  check_device_expressions<2>(transient);
}

TEST_CASE("Kokkos evaluates 3D flow residuals and Jacobians", "[assembly][kokkos][flow][device]") {
  const bool transient = GENERATE(false, true);
  check_device_expressions<3>(transient);
}
} // namespace

int main(int argc, char **argv) {
  Kokkos::ScopeGuard runtime(argc, argv);
  std::cout << "Kokkos execution space: " << Kokkos::DefaultExecutionSpace::name() << '\n';
  return Catch::Session().run(argc, argv);
}
