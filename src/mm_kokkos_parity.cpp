#include "mm_kokkos_parity.h"
#include "mm_kokkos_legacy.h"

#include "linalg/sparse_matrix_tpetra.h"
extern "C" {
#include "mm_as.h"
#include "rf_mp.h"
#include "rf_node_const.h"
#include "sl_util_structs.h"
}
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr double relative_tolerance = 1.e-9;
constexpr double scale_tolerance = 1.e-12;
constexpr double global_scale_fraction = 1.e-3; // Floor for rows with vanishing scales.
constexpr int reported_failures = 5;

bool all_ranks(bool valid) {
  int local = valid ? 1 : 0, global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, goma_comm);
  return global != 0;
}

int rank() {
  int value = 0;
  MPI_Comm_rank(goma_comm, &value);
  return value;
}

void report_failure(const std::string &message) {
  std::fprintf(stderr, "P_%d: Kokkos parity failed: %s\n", rank(), message.c_str());
  std::fflush(stderr);
}

// Comparison results for one kind of entry, split by Dirichlet-replaced rows.
struct Statistics {
  long long entries[2] = {0, 0};
  long long failures[2] = {0, 0};
  double ratio[2] = {0.0, 0.0}; // Largest |difference| / tolerance.
  double difference[2] = {0.0, 0.0};
  int printed = 0;

  void add(const char *kind,
           bool constrained,
           GO row,
           GO column,
           double legacy,
           double kokkos,
           double scale,
           double global_scale) {
    const int c = constrained ? 1 : 0;
    const double difference_ = std::abs(kokkos - legacy);
    const double tolerance = relative_tolerance * std::max(std::abs(kokkos), std::abs(legacy)) +
                             scale_tolerance * (scale + global_scale_fraction * global_scale);
    const double ratio_ = tolerance > 0.0      ? difference_ / tolerance
                          : difference_ == 0.0 ? 0.0
                                               : INFINITY;
    ++entries[c];
    difference[c] = std::max(difference[c], difference_);
    // NaN never passes.
    if (!(ratio_ <= 1.0)) {
      ++failures[c];
      ratio[c] = std::isnan(ratio_) ? INFINITY : std::max(ratio[c], ratio_);
      if (printed++ < reported_failures) {
        std::fprintf(stderr,
                     "P_%d: Kokkos parity %s row %lld column %lld%s: legacy %.17g kokkos %.17g "
                     "(scale %.3g)\n",
                     rank(), kind, static_cast<long long>(row), static_cast<long long>(column),
                     constrained ? " (constrained)" : "", legacy, kokkos, scale);
      }
    } else {
      ratio[c] = std::max(ratio[c], ratio_);
    }
  }

  void reduce() {
    MPI_Allreduce(MPI_IN_PLACE, entries, 2, MPI_LONG_LONG, MPI_SUM, goma_comm);
    MPI_Allreduce(MPI_IN_PLACE, failures, 2, MPI_LONG_LONG, MPI_SUM, goma_comm);
    MPI_Allreduce(MPI_IN_PLACE, ratio, 2, MPI_DOUBLE, MPI_MAX, goma_comm);
    MPI_Allreduce(MPI_IN_PLACE, difference, 2, MPI_DOUBLE, MPI_MAX, goma_comm);
  }
  long long total_failures() const { return failures[0] + failures[1]; }
};

// Residual-row scales from the last Jacobian fill, for later residual-only requests.
struct ResidualScales {
  const void *matrix = nullptr;
  std::vector<double> rows;
} cached_scales;

std::vector<char> constrained_rows(const Exo_DB &exo, std::size_t owned_rows) {
  std::vector<char> constrained(owned_rows, 0);
  for (int n = 0; Nodes != nullptr && n < exo.num_nodes; ++n) {
    const auto *node = Nodes[n];
    if (node == nullptr || node->DBC == nullptr || node->DBC[0] == nullptr ||
        node->Nodal_Vars_Info == nullptr || node->Nodal_Vars_Info[0] == nullptr) {
      continue;
    }
    for (int offset = 0; offset < node->Nodal_Vars_Info[0]->Num_Unknowns; ++offset) {
      const auto row = static_cast<std::size_t>(node->First_Unknown[0] + offset);
      if (node->DBC[0][offset] != -1 && row < owned_rows) {
        constrained[row] = 1;
      }
    }
  }
  return constrained;
}
} // namespace

extern "C" int goma_kokkos_parity_enabled(void) {
  const char *value = std::getenv("GOMA_KOKKOS_PARITY");
  int enabled = value != nullptr && std::string(value) == "1" ? 1 : 0;
  MPI_Bcast(&enabled, 1, MPI_INT, 0, goma_comm);
  return enabled;
}

extern "C" int goma_kokkos_parity_fill(struct GomaLinearSolverData *ams,
                                       const Exo_DB *exo,
                                       const double *x,
                                       const double *xdot,
                                       double delta_t,
                                       double theta,
                                       double *resid_vector,
                                       goma_kokkos_reference_fill legacy,
                                       void *data) {
  try {
    auto *matrix = ams != nullptr ? static_cast<GomaSparseMatrix>(ams->GomaMatrixData) : nullptr;
    const bool valid = af != nullptr && exo != nullptr && legacy != nullptr && matrix != nullptr &&
                       matrix->type == GOMA_SPARSE_MATRIX_TYPE_TPETRA && matrix->data != nullptr &&
                       matrix->n_rows >= 0 && matrix->n_cols >= matrix->n_rows &&
                       (matrix->n_cols == 0 || (x != nullptr && resid_vector != nullptr));
    if (!all_ranks(valid)) {
      report_failure("requires a Tpetra matrix and local solution and residual vectors");
      return -1;
    }
    const bool residual = af->Assemble_Residual != 0;
    const bool jacobian = af->Assemble_Jacobian != 0;
    auto &adapter = *static_cast<TpetraSparseMatrix *>(matrix->data);
    const auto owned = static_cast<std::size_t>(matrix->n_rows);
    const auto local = static_cast<std::size_t>(matrix->n_cols);
    auto copy_values = [&]() {
      const auto view = adapter.matrix->getLocalValuesHost(Tpetra::Access::ReadOnly);
      return std::vector<double>(view.data(), view.data() + view.extent(0));
    };
    auto restore_values = [&](const std::vector<double> &saved) {
      const auto view = adapter.matrix->getLocalValuesHost(Tpetra::Access::ReadWrite);
      std::copy(saved.begin(), saved.end(), view.data());
    };

    // Both fills start from the caller's state, which need not be zero.
    const std::vector<double> base_residual(resid_vector, resid_vector + local);
    std::vector<double> base_matrix, legacy_matrix;
    if (jacobian) {
      base_matrix = copy_values();
    }
    if (legacy(data) != 0) { // The legacy loop reduces its errors over all ranks.
      report_failure("legacy reference fill failed");
      return -1;
    }
    const std::vector<double> legacy_residual(resid_vector, resid_vector + local);
    if (jacobian) {
      legacy_matrix = copy_values();
      restore_values(base_matrix);
    }
    std::copy(base_residual.begin(), base_residual.end(), resid_vector);
    if (goma_kokkos_matrix_fill(ams, exo, x, xdot, delta_t, theta, resid_vector) != 0) {
      return -1;
    }

    const auto constrained = constrained_rows(*exo, owned);
    const auto graph = adapter.matrix->getCrsGraph()->getLocalGraphHost();
    const auto &column_map = *adapter.matrix->getColMap();
    Statistics residual_statistics, matrix_statistics;
    std::vector<double> scales(owned, 0.0);
    if (jacobian) {
      std::unordered_map<GO, std::size_t> local_of;
      for (std::size_t i = 0; i < local; ++i) {
        local_of.emplace(matrix->global_ids[i], i);
      }
      const auto kokkos = copy_values();
      std::vector<double> largest(owned, 0.0);
      for (std::size_t row = 0; row < owned; ++row) {
        for (auto k = graph.row_map(row); k < graph.row_map(row + 1); ++k) {
          const double entry = legacy_matrix[k] - base_matrix[k];
          largest[row] = std::max(largest[row], std::abs(entry));
          const auto found = local_of.find(column_map.getGlobalElement(graph.entries(k)));
          if (found != local_of.end()) {
            scales[row] += std::abs(entry * x[found->second]);
          }
        }
        scales[row] += std::abs(legacy_residual[row] - base_residual[row]);
      }
      double global = owned > 0 ? *std::max_element(largest.begin(), largest.end()) : 0.0;
      MPI_Allreduce(MPI_IN_PLACE, &global, 1, MPI_DOUBLE, MPI_MAX, goma_comm);
      for (std::size_t row = 0; row < owned; ++row) {
        for (auto k = graph.row_map(row); k < graph.row_map(row + 1); ++k) {
          matrix_statistics.add("matrix", constrained[row] != 0, matrix->global_ids[row],
                                column_map.getGlobalElement(graph.entries(k)),
                                legacy_matrix[k] - base_matrix[k], kokkos[k] - base_matrix[k],
                                largest[row], global);
        }
      }
      cached_scales = {matrix, scales};
    } else if (cached_scales.matrix == matrix && cached_scales.rows.size() == owned) {
      scales = cached_scales.rows;
    } else {
      double largest = 0.0;
      for (std::size_t row = 0; row < owned; ++row) {
        largest = std::max(largest, std::abs(legacy_residual[row] - base_residual[row]));
      }
      MPI_Allreduce(MPI_IN_PLACE, &largest, 1, MPI_DOUBLE, MPI_MAX, goma_comm);
      std::fill(scales.begin(), scales.end(), largest);
    }
    if (residual) {
      double global = owned > 0 ? *std::max_element(scales.begin(), scales.end()) : 0.0;
      MPI_Allreduce(MPI_IN_PLACE, &global, 1, MPI_DOUBLE, MPI_MAX, goma_comm);
      for (std::size_t row = 0; row < owned; ++row) {
        residual_statistics.add("residual", constrained[row] != 0, matrix->global_ids[row], -1,
                                legacy_residual[row] - base_residual[row],
                                resid_vector[row] - base_residual[row], scales[row], global);
      }
    }
    // External entries belong to their owners in both implementations.
    bool external = true;
    for (std::size_t i = owned; i < local; ++i) {
      external = external && resid_vector[i] == base_residual[i];
    }
    residual_statistics.reduce();
    matrix_statistics.reduce();
    const bool externals = all_ranks(external);
    if (rank() == 0) {
      std::fprintf(stderr, "Kokkos parity%s%s:", residual ? " R" : "", jacobian ? " J" : "");
      if (residual) {
        std::fprintf(stderr,
                     " residual rows %lld+%lld constrained, max ratio %.2e/%.2e, max diff "
                     "%.2e/%.2e;",
                     residual_statistics.entries[0], residual_statistics.entries[1],
                     residual_statistics.ratio[0], residual_statistics.ratio[1],
                     residual_statistics.difference[0], residual_statistics.difference[1]);
      }
      if (jacobian) {
        std::fprintf(stderr,
                     " matrix entries %lld+%lld constrained, max ratio %.2e/%.2e, max diff "
                     "%.2e/%.2e;",
                     matrix_statistics.entries[0], matrix_statistics.entries[1],
                     matrix_statistics.ratio[0], matrix_statistics.ratio[1],
                     matrix_statistics.difference[0], matrix_statistics.difference[1]);
      }
      std::fprintf(stderr, " %s\n",
                   residual_statistics.total_failures() + matrix_statistics.total_failures() == 0 &&
                           externals
                       ? "PASS"
                       : "FAIL");
      std::fflush(stderr);
    }
    if (!externals) {
      report_failure("Kokkos fill changed external residual entries");
      return -1;
    }
    if (residual_statistics.total_failures() + matrix_statistics.total_failures() != 0) {
      report_failure(std::to_string(residual_statistics.total_failures()) + " residual and " +
                     std::to_string(matrix_statistics.total_failures()) +
                     " matrix entries outside tolerance");
      return -1;
    }
    return 0;
  } catch (const std::exception &error) {
    report_failure(error.what());
    return -1;
  } catch (...) {
    report_failure("unknown exception");
    return -1;
  }
}
