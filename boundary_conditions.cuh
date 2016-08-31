/*
 * boundary_conditions.cuh
 *
 *  Created on: Jul 5, 2016
 *      Author: wortiz
 */

#ifndef BOUNDARY_CONDITIONS_CUH_
#define BOUNDARY_CONDITIONS_CUH_

#include "ns_constants.h"

#include "ns_structs.h"


void set_boundary_condition(boundary_condition *bc, int dirichlet_index,
                            double value, boundary_condition_type type,
                            equation eqn);

#endif /* BOUNDARY_CONDITIONS_CUH_ */
