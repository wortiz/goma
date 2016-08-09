/*
 * cuda_constants.cuh
 *
 *  Created on: Aug 9, 2016
 *      Author: wortiz
 */

#ifndef CUDA_CONSTANTS_CUH_
#define CUDA_CONSTANTS_CUH_

namespace cuda {

enum element_type {
  BILINEAR_QUAD,
  BIQUAD_QUAD
};

enum shape {
  QUADRILATERAL
};

enum equation {
  CUDA_VELOCITY1 = 0,
  CUDA_VELOCITY2,
  CUDA_PRESSURE,
  CUDA_NUM_EQNS
};

enum boundary_condition_type {
  CUDA_DIRICHLET,
  CUDA_PARABOLIC
};

} // namespace cuda

#define CUDA_MDE 9

#define CUDA_DIM 2

#define MAX_BCS 10

#endif /* CUDA_CONSTANTS_CUH_ */
