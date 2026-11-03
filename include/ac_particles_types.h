/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) 2026 Goma Developers, National Technology & Engineering   *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

/*
 * Particle types needed by GomaContext. Kept free of other Goma headers so
 * that util/goma_context.h can include it.
 */

#ifndef GOMA_AC_PARTICLES_TYPES_H
#define GOMA_AC_PARTICLES_TYPES_H

#define MAX_PBC_REAL_VALUES                 5
#define MAX_PBC_INT_VALUES                  5
#define MAX_DOMAIN_REAL_VALUES              6
#define MAX_PARTICLE_MODEL_DATA_VALUES      7
#define MAX_PBC_STRING_DATA_LENGTH          80
#define MAX_PARTICLE_OUTPUT_VARIABLE_LENGTH 12
#define MAX_PARTICLE_FILENAME_LENGTH        80
#define MAX_PARTICLE_STRING_LENGTH          255

#define XI_BOUNDARY_TOLERANCE0 1.0e-8
#define XI_BOUNDARY_TOLERANCE1 1.0e-12
#define XI_BOUNDARY_TOLERANCE2 1.0e-14

enum Particle_Model_t {
  NO_PARTICLE_MODEL,
  TRACER_EXPLICIT,
  TRACER_IMPLICIT,
  INERTIAL_TRACER_EXPLICIT,
  INERTIAL_TRACER_IMPLICIT,
  SWIMMER_EXPLICIT,
  SWIMMER_IMPLICIT,
  CHARGED_TRACER_EXPLICIT,
  CHARGED_TRACER_IMPLICIT,
  DIELECTROPHORETIC_TRACER_IMPLICIT
};

enum Particle_Output_Format_t { FLAT_TEXT, TECPLOT };

enum PBC { PBC_OUTFLOW, PBC_SOURCE, PBC_TARGET, PBC_FREESTREAM_SOURCE, PBC_IMPERMEABLE };

enum Particle_Domain_t { UNRESTRICTED, BRICK, ACIS_OBJECT };

typedef struct {
  enum PBC type;
  int SS_id;
  double real_data[MAX_PBC_REAL_VALUES];
  int int_data[MAX_PBC_INT_VALUES];
  char string_data[MAX_PBC_STRING_DATA_LENGTH];
} PBC_t;

typedef struct element_particle_info_t {
  int *PBC_side_id;
#ifdef PARALLEL
  int *owner_local_element_id; /* On other side of face on processor boundary, what's the owner's
                                  local element id? */
  int *ghost_proc; /* What processors and their local element id's do these particles need to be
                      sent to? */
  int *ghost_local_elem_id;
  int num_ghost_target_elems;
#endif
  double *source_term;
  int *list_PBC_IMPERMEABLE;
} element_particle_info_t;

typedef char particle_variable_s[MAX_PARTICLE_OUTPUT_VARIABLE_LENGTH];
typedef char particle_filename_s[MAX_PARTICLE_FILENAME_LENGTH];
typedef char particle_s[MAX_PARTICLE_STRING_LENGTH];

#endif /* GOMA_AC_PARTICLES_TYPES_H */
