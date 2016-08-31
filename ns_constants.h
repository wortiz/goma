/*
 * constants.h
 *
 *  Created on: Jun 30, 2016
 *      Author: wortiz
 */

#ifndef NS_CONSTANTS_H_
#define NS_CONSTANTS_H_

enum element_type {
  BILINEAR_QUAD,
  BIQUAD_QUAD
};

enum shape {
  QUADRILATERAL
};

enum equation {
  VELOCITY1 = 0,
  VELOCITY2,
  PRESSURE,
  AUX_VELOCITY1,
  AUX_VELOCITY2,
  AUX_PRESSURE,
  NUM_EQNS
};


enum boundary_condition_type {
  DIRICHLET,
  PARABOLIC
};

#define MDE 9

#define DIM 2

// Max bcs for a given element
#define MAX_BCS 16

#endif /* NS_CONSTANTS_H_ */
