/*
 * boundary_conditions.cu
 *
 *  Created on: Jul 5, 2016
 *      Author: wortiz
 */


#include "boundary_conditions.cuh"

void set_boundary_condition(boundary_condition *bc, int dirichlet_index,
                            double value, boundary_condition_type type,
                            equation eqn)
{
  bc->dirichlet_index = dirichlet_index;
  bc->value = value;
  bc->type = type;
  bc->eqn = eqn;
}
