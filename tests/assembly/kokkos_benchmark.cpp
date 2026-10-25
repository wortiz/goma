/* Kernel benchmark for Q2/Q1 geometry, element, and Dirichlet evaluation.
 *
 * Runs the same distorted structured mesh on Kokkos::Serial and, when it differs, on the
 * default device (CUDA in GPU builds), reports kernel and host/device transfer times
 * separately, and checks that every device reproduces the Serial residuals and Jacobians.
 * Tpetra scatter and linear solves are not included.
 */
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mm_kokkos_constraints.hpp"

namespace {
using namespace goma::assembly;

struct Options {
  int dim = 2;
  int cells = 0; // Per direction; 0 selects a dimension-dependent default.
  int repeats = 5;
  bool transient = true;
  double tolerance = 1e-10;
};

struct Timing {
  std::string name;
  std::vector<double> samples;
  double best() const { return *std::min_element(samples.begin(), samples.end()); }
  double median() const {
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    return sorted[sorted.size() / 2];
  }
};

template <class Function> double timed(Function &&function) {
  Kokkos::fence();
  Kokkos::Timer timer;
  function();
  Kokkos::fence();
  return timer.seconds();
}

template <int Dim, class Device> struct Problem {
  using Topology = Q2Q1<Dim>;
  ElementCoordinatesOn<Device> coordinates;
  ElementFieldsOn<Device> solution;
  ElementFieldsOn<Device> velocity_dot;
  ElementConstraintsOn<Device> constraints;
  DeviceView<ConstantFluid<Dim> *, Device> materials;
  DeviceView<FlowExpressions<Dim> *, Device> expressions;

  explicit Problem(int elements)
      : coordinates("benchmark coordinates", elements, Topology::velocity_nodes, Dim),
        solution("benchmark solution", elements, Topology::dofs),
        velocity_dot("benchmark velocity derivative", elements, Topology::velocity_dofs),
        constraints("benchmark constraints", elements, Topology::dofs),
        materials("benchmark materials", elements), expressions("benchmark expressions", elements) {
  }
};

// Host data for a unit-cube mesh of cells^Dim elements with a smooth interior distortion.
template <int Dim> struct HostData {
  using Topology = Q2Q1<Dim>;
  using Host = Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>;
  Problem<Dim, Host> data;

  HostData(int cells, bool transient) : data(elements(cells)) {
    constexpr double pi = 3.14159265358979323846;
    const double h = 1.0 / cells;
    for (int e = 0; e < elements(cells); ++e) {
      int cell[Dim];
      for (int d = 0, rest = e; d < Dim; ++d, rest /= cells) {
        cell[d] = rest % cells;
      }
      data.materials(e).density = 1.3;
      data.materials(e).viscosity = 0.05;
      data.expressions(e).transient = transient;
      data.expressions(e).continuity = {true, 1.0};
      for (int a = 0; a < Dim; ++a) {
        data.materials(e).source[a] = a == 0 ? 0.5 : 0.0;
        data.expressions(e).momentum[a] = {{transient, 1.0}, {true, 1.0}, {true, 1.0}, {true, 1.0}};
      }
      for (int node = 0; node < Topology::velocity_nodes; ++node) {
        double reference[Dim], x[Dim];
        for (int d = 0; d < Dim; ++d) {
          reference[d] = (cell[d] + 0.5 * (Topology::node_coordinate(node, d) + 1)) * h;
        }
        for (int d = 0; d < Dim; ++d) {
          // Vanishes on the boundary; small enough to keep every Jacobian positive.
          double bump = 0.05 * h;
          for (int k = 0; k < Dim; ++k) {
            bump *= std::sin(pi * reference[k]);
          }
          x[d] = reference[d] + bump;
          data.coordinates(e, node, d) = x[d];
        }
        for (int a = 0; a < Dim; ++a) {
          const int dof = a * Topology::velocity_nodes + node;
          data.solution(e, dof) = std::sin(pi * x[0] + a) * std::cos(pi * x[Dim - 1]);
          data.velocity_dot(e, dof) = 0.1 * std::cos(pi * x[a]);
          // Hard-set velocity on the x = 0 face.
          if (cell[0] == 0 && Topology::node_coordinate(node, 0) == -1) {
            data.constraints(e, dof) = {DirichletMode::hard_set, data.solution(e, dof)};
          }
        }
      }
      for (int node = 0; node < Topology::pressure_nodes; ++node) {
        data.solution(e, Topology::velocity_dofs + node) = 1.0 + 0.1 * node;
      }
    }
    // Residual-mode pressure datum on the first element's first corner.
    data.constraints(0, Topology::velocity_dofs) = {DirichletMode::residual, 1.0};
  }

  static int elements(int cells) {
    int count = 1;
    for (int d = 0; d < Dim; ++d) {
      count *= cells;
    }
    return count;
  }
};

template <int Dim> struct Results {
  std::vector<double> residual, jacobian;
};

template <int Dim, class Device>
Results<Dim>
run(const HostData<Dim> &host, const Options &options, std::vector<Timing> &timings) {
  const int elements = static_cast<int>(host.data.solution.extent(0));
  Problem<Dim, Device> problem(elements);
  // Device layouts may differ from the host data (CUDA uses LayoutLeft), so stage through
  // mirrors. Problem data is copied once; fields are refreshed for every evaluation.
  auto upload = [](const auto &device, const auto &source) {
    auto mirror = Kokkos::create_mirror_view(device);
    Kokkos::deep_copy(mirror, source);
    Kokkos::deep_copy(device, mirror);
    return mirror;
  };
  upload(problem.coordinates, host.data.coordinates);
  upload(problem.constraints, host.data.constraints);
  upload(problem.materials, host.data.materials);
  upload(problem.expressions, host.data.expressions);
  const auto solution = upload(problem.solution, host.data.solution);
  const auto velocity_dot = upload(problem.velocity_dot, host.data.velocity_dot);
  timings = {{"geometry setup", {}},  {"field upload", {}},       {"residual", {}},
             {"jacobian", {}},        {"residual+jacobian", {}}, {"result download", {}}};
  ElementGeometry<Dim, Device> geometry;
  ElementResult<Dim, Device> result;
  const double coefficient = options.transient ? 3.0 / 0.05 : 0.0;
  auto evaluate = [&](EvaluationRequest request) {
    result = evaluate_q2q1_elements(geometry, problem.solution, problem.velocity_dot,
                                    problem.materials, problem.expressions, coefficient, request);
    apply_q2q1_dirichlet(result, problem.solution, problem.constraints, request, 1.e15);
  };
  // One untimed pass absorbs first-launch and allocation warm-up.
  for (int pass = 0; pass <= options.repeats; ++pass) {
    const bool record = pass > 0;
    const double times[] = {
        timed([&] { geometry = prepare_q2q1_geometry<Dim, Device>(problem.coordinates); }),
        timed([&] {
          Kokkos::deep_copy(problem.solution, solution);
          Kokkos::deep_copy(problem.velocity_dot, velocity_dot);
        }),
        timed([&] { evaluate(EvaluationRequest::residual); }),
        timed([&] { evaluate(EvaluationRequest::jacobian); }),
        timed([&] { evaluate(EvaluationRequest::both); }),
        timed([&] {
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.residual);
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.jacobian);
        })};
    for (std::size_t i = 0; record && i < timings.size(); ++i) {
      timings[i].samples.push_back(times[i]);
    }
  }
  const auto status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.status);
  const auto geometry_status =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), geometry.status);
  for (int e = 0; e < elements; ++e) {
    if (status(e) != 0 || geometry_status(e) != 0) {
      std::fprintf(stderr, "element %d failed: geometry status %d, element status %d\n", e,
                   geometry_status(e), status(e));
      std::exit(EXIT_FAILURE);
    }
  }
  const auto residual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.residual);
  const auto jacobian = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.jacobian);
  // Flatten in a layout-independent order for comparison across devices.
  Results<Dim> output;
  for (int e = 0; e < elements; ++e) {
    for (int row = 0; row < Q2Q1<Dim>::dofs; ++row) {
      output.residual.push_back(residual(e, row));
      for (int column = 0; column < Q2Q1<Dim>::dofs; ++column) {
        output.jacobian.push_back(jacobian(e, row, column));
      }
    }
  }
  return output;
}

// Largest difference relative to the magnitude of its block (an element residual or a
// Jacobian row), so penalty rows do not hide errors elsewhere.
double relative_difference(const std::vector<double> &a,
                           const std::vector<double> &b,
                           std::size_t block) {
  double worst = 0.0;
  for (std::size_t start = 0; start < a.size(); start += block) {
    double scale = 0.0, difference = 0.0;
    for (std::size_t i = start; i < start + block; ++i) {
      scale = std::max(scale, std::abs(a[i]));
      difference = std::max(difference, std::abs(a[i] - b[i]));
    }
    worst = std::max(worst, difference / std::max(scale, 1e-300));
  }
  return worst;
}

template <int Dim> int benchmark(const Options &options) {
  using Topology = Q2Q1<Dim>;
  using Serial = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;
  const int cells = options.cells > 0 ? options.cells : (Dim == 2 ? 256 : 16);
  const HostData<Dim> host(cells, options.transient);
  const int elements = HostData<Dim>::elements(cells);
  const double jacobian_mb = 8.0 * elements * Topology::dofs * Topology::dofs / 1048576.0;
  std::printf("Q2/Q1 %dD %s: %d^%d = %d elements, %d DOFs per element, %.1f MiB of element "
              "Jacobians, best/median of %d repeats\n",
              Dim, options.transient ? "transient" : "steady", cells, Dim, elements,
              Topology::dofs, jacobian_mb, options.repeats);

  std::vector<Timing> serial_timings, device_timings;
  const auto reference = run<Dim, Serial>(host, options, serial_timings);
  std::vector<std::string> names = {Kokkos::Serial::name()};
  std::vector<std::vector<Timing>> all = {serial_timings};
  bool agree = true;
  if constexpr (!std::is_same_v<DefaultDevice, Serial>) {
    const auto device = run<Dim, DefaultDevice>(host, options, device_timings);
    names.push_back(DefaultDevice::execution_space::name());
    all.push_back(device_timings);
    const double residual =
        relative_difference(reference.residual, device.residual, Topology::dofs);
    const double jacobian =
        relative_difference(reference.jacobian, device.jacobian, Topology::dofs);
    agree = residual <= options.tolerance && jacobian <= options.tolerance;
    std::printf("%s vs Serial max blockwise relative difference: residual %.2e, Jacobian %.2e "
                "(tolerance %.1e): %s\n",
                names.back().c_str(), residual, jacobian, options.tolerance,
                agree ? "PASS" : "FAIL");
  }

  std::printf("\n%-20s", "phase (ms)");
  for (const auto &name : names) {
    std::printf("  %10s best  %10s median", name.c_str(), name.c_str());
  }
  if (names.size() > 1) {
    std::printf("  %10s", "speedup");
  }
  std::printf("\n");
  for (std::size_t phase = 0; phase < serial_timings.size(); ++phase) {
    std::printf("%-20s", serial_timings[phase].name.c_str());
    for (const auto &timings : all) {
      std::printf("  %15.3f  %17.3f", 1e3 * timings[phase].best(), 1e3 * timings[phase].median());
    }
    if (all.size() > 1) {
      std::printf("  %9.1fx", all[0][phase].best() / all[1][phase].best());
    }
    std::printf("\n");
  }
  std::printf("\n%-20s", "elements/s (R+J)");
  for (const auto &timings : all) {
    std::printf("  %15.3e  %17s", elements / timings[4].best(), "");
  }
  std::printf("\n");
  return agree ? EXIT_SUCCESS : EXIT_FAILURE;
}

void usage(const char *program) {
  std::fprintf(stderr,
               "usage: %s [--dim 2|3] [--cells N] [--repeats N] [--steady] [--tolerance T]\n"
               "  Times Q2/Q1 geometry, element, and Dirichlet kernels on Kokkos::Serial and\n"
               "  the default device, and checks that both produce the same element data.\n",
               program);
}
} // namespace

int main(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool value = i + 1 < argc;
    if (arg == "--dim" && value) {
      options.dim = std::atoi(argv[++i]);
    } else if (arg == "--cells" && value) {
      options.cells = std::atoi(argv[++i]);
    } else if (arg == "--repeats" && value) {
      options.repeats = std::atoi(argv[++i]);
    } else if (arg == "--tolerance" && value) {
      options.tolerance = std::atof(argv[++i]);
    } else if (arg == "--steady") {
      options.transient = false;
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else if (arg.rfind("--kokkos", 0) != 0) {
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }
  if ((options.dim != 2 && options.dim != 3) || options.cells < 0 || options.repeats < 1) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }
  Kokkos::ScopeGuard runtime(argc, argv);
  return options.dim == 2 ? benchmark<2>(options) : benchmark<3>(options);
}
