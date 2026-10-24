/* Q2/Q1 tensor bases in Goma's element-node and quadrature ordering. */
#ifndef GOMA_MM_KOKKOS_BASIS_HPP
#define GOMA_MM_KOKKOS_BASIS_HPP

#include "mm_kokkos_flow.hpp"

namespace goma::assembly {

template <int Dim> struct Q2Q1 {
  static_assert(Dim == 2 || Dim == 3);
  static constexpr int velocity_nodes = Dim == 2 ? 9 : 27;
  static constexpr int pressure_nodes = Dim == 2 ? 4 : 8;
  static constexpr int quadrature_points = velocity_nodes;
  static constexpr int velocity_dofs = Dim * velocity_nodes;
  static constexpr int dofs = velocity_dofs + pressure_nodes;

  // Corners first; hex node 20 is the center, followed by the six face nodes.
  KOKKOS_INLINE_FUNCTION static int node_coordinate(int node, int direction) {
    if constexpr (Dim == 2) {
      constexpr int nodes[9][2] = {{-1, -1}, {1, -1}, {1, 1},  {-1, 1}, {0, -1},
                                   {1, 0},   {0, 1},  {-1, 0}, {0, 0}};
      return nodes[node][direction];
    } else {
      constexpr int nodes[27][3] = {
          {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, -1, 1}, {1, -1, 1},  {1, 1, 1},
          {-1, 1, 1},   {0, -1, -1}, {1, 0, -1}, {0, 1, -1},  {-1, 0, -1}, {-1, -1, 0}, {1, -1, 0},
          {1, 1, 0},    {-1, 1, 0},  {0, -1, 1}, {1, 0, 1},   {0, 1, 1},   {-1, 0, 1},  {0, 0, 0},
          {0, 0, -1},   {0, 0, 1},   {-1, 0, 0}, {1, 0, 0},   {0, -1, 0},  {0, 1, 0}};
      return nodes[node][direction];
    }
  }

  template <int Degree>
  KOKKOS_INLINE_FUNCTION static Basis<Dim> basis(int node, const double xi[Dim]) {
    static_assert(Degree == 1 || Degree == 2);
    double value[Dim];
    double derivative[Dim];
    for (int d = 0; d < Dim; ++d) {
      const int coordinate = node_coordinate(node, d);
      if constexpr (Degree == 1) {
        value[d] = 0.5 * (1.0 + coordinate * xi[d]);
        derivative[d] = 0.5 * coordinate;
      } else {
        if (coordinate == 0) {
          value[d] = 1.0 - xi[d] * xi[d];
          derivative[d] = -2.0 * xi[d];
        } else {
          value[d] = 0.5 * xi[d] * (xi[d] + coordinate);
          derivative[d] = xi[d] + 0.5 * coordinate;
        }
      }
    }
    Basis<Dim> result;
    result.value = 1.0;
    for (int d = 0; d < Dim; ++d) {
      result.value *= value[d];
      result.gradient[d] = derivative[d];
      for (int r = 0; r < Dim; ++r) {
        if (r != d) {
          result.gradient[d] *= value[r];
        }
      }
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION static double quadrature(int point, double xi[Dim]) {
    constexpr double location = 0.77459666924148340428;
    double weight = 1.0;
    for (int d = 0; d < Dim; ++d) {
      const int index = point % 3;
      point /= 3;
      xi[d] = index == 0 ? location : (index == 1 ? 0.0 : -location);
      weight *= index == 1 ? 8.0 / 9.0 : 5.0 / 9.0;
    }
    return weight;
  }
};

} // namespace goma::assembly
#endif
