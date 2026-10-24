/* Fixed-mesh Cartesian flow expressions. Legacy globals belong in the host adapter. */
#ifndef GOMA_MM_KOKKOS_FLOW_HPP
#define GOMA_MM_KOKKOS_FLOW_HPP

#include <Kokkos_Macros.hpp>

namespace goma::assembly {

// Match fv->grad_v: gradient[direction][velocity component].
template <typename Scalar, int Dim> struct FlowState {
  static_assert(Dim == 2 || Dim == 3);
  Scalar velocity[Dim]{};
  Scalar gradient[Dim][Dim]{};
  Scalar velocity_dot[Dim]{};
  Scalar pressure{};
};

template <int Dim> struct Basis {
  double value = 0.0;
  double gradient[Dim]{};
};

template <typename Scalar, int Dim> struct MaterialValues {
  Scalar density{};
  Scalar viscosity{};
  Scalar source[Dim]{}; // Force per volume, as for Momentum Source = CONSTANT.
};

template <int Dim> struct ConstantFluid {
  double density = 0.0;
  double viscosity = 0.0;
  double source[Dim]{};

  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION MaterialValues<Scalar, Dim>
  evaluate(const FlowState<Scalar, Dim> &) const {
    MaterialValues<Scalar, Dim> result;
    result.density = density;
    result.viscosity = viscosity;
    for (int d = 0; d < Dim; ++d) {
      result.source[d] = source[d];
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION MaterialValues<double, Dim>
  linearize(const FlowState<double, Dim> &, const FlowState<double, Dim> &) const {
    return {};
  }
};

// A velocity DOF perturbs both u and u_dot by the solver's time coefficient.
// The caller supplies that coefficient; expressions do not choose a time scheme.
template <int Dim>
KOKKOS_INLINE_FUNCTION FlowState<double, Dim>
velocity_increment(const Basis<Dim> &trial, int component, double time_coefficient) {
  FlowState<double, Dim> result;
  result.velocity[component] = trial.value;
  result.velocity_dot[component] = time_coefficient * trial.value;
  for (int d = 0; d < Dim; ++d) {
    result.gradient[d][component] = trial.gradient[d];
  }
  return result;
}

template <int Dim>
KOKKOS_INLINE_FUNCTION FlowState<double, Dim> pressure_increment(const Basis<Dim> &trial) {
  FlowState<double, Dim> result;
  result.pressure = trial.value;
  return result;
}

template <int Dim> struct MomentumMass {
  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION Scalar residual(const FlowState<Scalar, Dim> &state,
                                         const MaterialValues<Scalar, Dim> &material,
                                         const Basis<Dim> &test,
                                         int component) const {
    return -test.value * material.density * state.velocity_dot[component];
  }

  KOKKOS_INLINE_FUNCTION double linearize(const FlowState<double, Dim> &state,
                                          const FlowState<double, Dim> &increment,
                                          const MaterialValues<double, Dim> &material,
                                          const MaterialValues<double, Dim> &tangent,
                                          const Basis<Dim> &test,
                                          int component) const {
    return -test.value * (tangent.density * state.velocity_dot[component] +
                          material.density * increment.velocity_dot[component]);
  }
};

template <int Dim> struct MomentumAdvection {
  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION Scalar residual(const FlowState<Scalar, Dim> &state,
                                         const MaterialValues<Scalar, Dim> &material,
                                         const Basis<Dim> &test,
                                         int component) const {
    Scalar convection = 0.0;
    for (int d = 0; d < Dim; ++d) {
      convection += state.velocity[d] * state.gradient[d][component];
    }
    return -test.value * material.density * convection;
  }

  KOKKOS_INLINE_FUNCTION double linearize(const FlowState<double, Dim> &state,
                                          const FlowState<double, Dim> &increment,
                                          const MaterialValues<double, Dim> &material,
                                          const MaterialValues<double, Dim> &tangent,
                                          const Basis<Dim> &test,
                                          int component) const {
    double result = 0.0;
    for (int d = 0; d < Dim; ++d) {
      result += tangent.density * state.velocity[d] * state.gradient[d][component] +
                material.density * (increment.velocity[d] * state.gradient[d][component] +
                                    state.velocity[d] * increment.gradient[d][component]);
    }
    return -test.value * result;
  }
};

template <int Dim> struct MomentumStress {
  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION Scalar residual(const FlowState<Scalar, Dim> &state,
                                         const MaterialValues<Scalar, Dim> &material,
                                         const Basis<Dim> &test,
                                         int component) const {
    Scalar result = 0.0;
    for (int d = 0; d < Dim; ++d) {
      // Pi = -p I + mu (grad u + grad u^T); Goma uses -grad(test) : Pi.
      Scalar stress =
          material.viscosity * (state.gradient[d][component] + state.gradient[component][d]);
      if (d == component) {
        stress -= state.pressure;
      }
      result -= test.gradient[d] * stress;
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION double linearize(const FlowState<double, Dim> &state,
                                          const FlowState<double, Dim> &increment,
                                          const MaterialValues<double, Dim> &material,
                                          const MaterialValues<double, Dim> &tangent,
                                          const Basis<Dim> &test,
                                          int component) const {
    double result = 0.0;
    for (int d = 0; d < Dim; ++d) {
      double stress =
          tangent.viscosity * (state.gradient[d][component] + state.gradient[component][d]) +
          material.viscosity *
              (increment.gradient[d][component] + increment.gradient[component][d]);
      if (d == component) {
        stress -= increment.pressure;
      }
      result -= test.gradient[d] * stress;
    }
    return result;
  }
};

template <int Dim> struct MomentumSource {
  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION Scalar residual(const FlowState<Scalar, Dim> &,
                                         const MaterialValues<Scalar, Dim> &material,
                                         const Basis<Dim> &test,
                                         int component) const {
    return test.value * material.source[component];
  }

  KOKKOS_INLINE_FUNCTION double linearize(const FlowState<double, Dim> &,
                                          const FlowState<double, Dim> &,
                                          const MaterialValues<double, Dim> &,
                                          const MaterialValues<double, Dim> &tangent,
                                          const Basis<Dim> &test,
                                          int component) const {
    return test.value * tangent.source[component];
  }
};

template <int Dim> struct Continuity {
  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION Scalar residual(const FlowState<Scalar, Dim> &state,
                                         const Basis<Dim> &test) const {
    Scalar divergence = 0.0;
    for (int d = 0; d < Dim; ++d) {
      divergence += state.gradient[d][d];
    }
    return test.value * divergence;
  }

  KOKKOS_INLINE_FUNCTION double linearize(const FlowState<double, Dim> &increment,
                                          const Basis<Dim> &test) const {
    return residual(increment, test);
  }
};

struct Term {
  bool enabled = false;
  double multiplier = 1.0;
  KOKKOS_INLINE_FUNCTION bool active() const { return enabled && multiplier != 0.0; }
};

struct MomentumTerms {
  Term mass;
  Term advection;
  Term stress;
  Term source;
};

// Compose terms at a quadrature point; element scheduling and scatter are separate.
// No derivative storage is needed by residual evaluation, and linearization does
// not evaluate a residual. All geometry is fixed, with Cartesian vector bases.
template <int Dim> struct FlowExpressions {
  bool transient = false;
  MomentumTerms momentum[Dim];
  Term continuity;

  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION Scalar momentum_residual(const FlowState<Scalar, Dim> &state,
                                                  const MaterialValues<Scalar, Dim> &material,
                                                  const Basis<Dim> &test,
                                                  int component,
                                                  double weight) const {
    const auto &terms = momentum[component];
    Scalar result = 0.0;
    if (transient && terms.mass.active()) {
      result +=
          terms.mass.multiplier * MomentumMass<Dim>{}.residual(state, material, test, component);
    }
    if (terms.advection.active()) {
      result += terms.advection.multiplier *
                MomentumAdvection<Dim>{}.residual(state, material, test, component);
    }
    if (terms.stress.active()) {
      result += terms.stress.multiplier *
                MomentumStress<Dim>{}.residual(state, material, test, component);
    }
    if (terms.source.active()) {
      result += terms.source.multiplier *
                MomentumSource<Dim>{}.residual(state, material, test, component);
    }
    return weight * result;
  }

  KOKKOS_INLINE_FUNCTION double momentum_linearization(const FlowState<double, Dim> &state,
                                                       const FlowState<double, Dim> &increment,
                                                       const MaterialValues<double, Dim> &material,
                                                       const MaterialValues<double, Dim> &tangent,
                                                       const Basis<Dim> &test,
                                                       int component,
                                                       double weight) const {
    const auto &terms = momentum[component];
    double result = 0.0;
    if (transient && terms.mass.active()) {
      result += terms.mass.multiplier *
                MomentumMass<Dim>{}.linearize(state, increment, material, tangent, test, component);
    }
    if (terms.advection.active()) {
      result +=
          terms.advection.multiplier *
          MomentumAdvection<Dim>{}.linearize(state, increment, material, tangent, test, component);
    }
    if (terms.stress.active()) {
      result += terms.stress.multiplier * MomentumStress<Dim>{}.linearize(
                                              state, increment, material, tangent, test, component);
    }
    if (terms.source.active()) {
      result += terms.source.multiplier * MomentumSource<Dim>{}.linearize(
                                              state, increment, material, tangent, test, component);
    }
    return weight * result;
  }

  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION Scalar continuity_residual(const FlowState<Scalar, Dim> &state,
                                                    const Basis<Dim> &test,
                                                    double weight) const {
    if (!continuity.active()) {
      return Scalar(0.0);
    }
    return weight * continuity.multiplier * Continuity<Dim>{}.residual(state, test);
  }

  KOKKOS_INLINE_FUNCTION double continuity_linearization(const FlowState<double, Dim> &increment,
                                                         const Basis<Dim> &test,
                                                         double weight) const {
    if (!continuity.active()) {
      return 0.0;
    }
    return weight * continuity.multiplier * Continuity<Dim>{}.linearize(increment, test);
  }
};

} // namespace goma::assembly
#endif
