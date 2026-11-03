/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) 2022 Goma Developers, National Technology & Engineering   *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

/* mm_input_particles.c -- read in parameters for particles.
 *
 * Author: Matthew M. Hopkins
 * Created: 11/01/2001
 */

/*
 * $Id: mm_input_particles.c,v 5.1 2007-09-18 18:53:45 prschun Exp $
 */

/* Standard include files */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ac_particles.h"
#include "mm_eh.h"
#include "mm_input.h"
#include "rf_allo.h"
#include "rf_fem.h"
#include "rf_fem_const.h"
#include "std.h"

#define SLEN   256
#define SLEN_2 512

void rd_particle_specs(FILE *ifp, char *input) {
  PBC_t *PBC;
  char err_msg[MAX_CHAR_ERR_MSG], s_tmp[SLEN], s_tmp_save[SLEN_2], *s_ptr1, *s_ptr2;
  int i, j, iread;

  iread = look_for_optional(ifp, "Particle Specifications", input, '=');
  if (iread != 1) {
    goma_ctx->particles.dynamics = 0;
    goma_ctx->particles.number_sample_types = 0;
    goma_ctx->particles.number_samples_existing = NULL;
    goma_ctx->particles.number_samples = NULL;
    goma_ctx->particles.number_output_variables = NULL;
    goma_ctx->particles.output_variables = NULL;
    goma_ctx->particles.filename_template = NULL;
    goma_ctx->particles.num_pbcs = 0;
    goma_ctx->particles.pbcs = NULL;
    return;
  }

  printf("\n--- Particle specifications ---\n");
  goma_ctx->particles.dynamics = 1; /* turn on particle dynamics */

  /* What kind of particle model do you want?  This determines all
   * kinds of things, such as FEM<->particle coupling, particle
   * trajectory calculation, etc. */
  look_for(ifp, "Particle model", input, '=');
  if (!fgets(s_tmp, SLEN - 1, ifp))
    GOMA_EH(GOMA_ERROR, "Error reading Particle model card.");
  strip(s_tmp); /* s_tmp still has '\n' at end */
  if (s_tmp[strlen(s_tmp) - 1] == '\n')
    s_tmp[strlen(s_tmp) - 1] = 0;
  strcpy(s_tmp_save, s_tmp);
  s_ptr1 = strpbrk(s_tmp_save, " \t\n");
  if (s_ptr1 != NULL)
    s_ptr1[0] = 0;
  if (!strncmp("INERTIAL_TRACER_EXPLICIT", s_tmp, 24))
    goma_ctx->particles.model = INERTIAL_TRACER_EXPLICIT;
  else if (!strncmp("INERTIAL_TRACER_IMPLICIT", s_tmp, 24))
    goma_ctx->particles.model = INERTIAL_TRACER_IMPLICIT;
  else if (!strncmp("TRACER_EXPLICIT", s_tmp, 15))
    goma_ctx->particles.model = TRACER_EXPLICIT;
  else if (!strncmp("TRACER_IMPLICIT", s_tmp, 15))
    goma_ctx->particles.model = TRACER_IMPLICIT;
  else if (!strncmp("SWIMMER_EXPLICIT", s_tmp, 16))
    goma_ctx->particles.model = SWIMMER_EXPLICIT;
  else if (!strncmp("SWIMMER_IMPLICIT", s_tmp, 16))
    goma_ctx->particles.model = SWIMMER_IMPLICIT;
  else if (!strncmp("CHARGED_TRACER_EXPLICIT", s_tmp, 23))
    goma_ctx->particles.model = CHARGED_TRACER_EXPLICIT;
  else if (!strncmp("CHARGED_TRACER_IMPLICIT", s_tmp, 23))
    goma_ctx->particles.model = CHARGED_TRACER_IMPLICIT;
  else if (!strncmp("DIELECTROPHORETIC_TRACER_IMPLICIT", s_tmp, 33))
    goma_ctx->particles.model = DIELECTROPHORETIC_TRACER_IMPLICIT;
  else {
    sprintf(s_tmp_save, "Unknown Particle model: %s\n", s_tmp);
    GOMA_EH(GOMA_ERROR, s_tmp_save);
  }
  printf("Setting Particle_Model = %s\n", s_tmp_save);

  switch (goma_ctx->particles.model) {
  case INERTIAL_TRACER_IMPLICIT:
  case INERTIAL_TRACER_EXPLICIT:
    s_ptr1 = s_tmp + 24;
    if (!(s_ptr2 = strtok(s_ptr1, " \t\n"))) {
      sprintf(err_msg, "Error reading random walk coefficient for %s particle model.", s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
    goma_ctx->particles.model_data[0] = atof(s_ptr2);
    printf("Setting random walk coefficient = %g\n", goma_ctx->particles.model_data[0]);

    if (!(s_ptr2 = strtok(NULL, " \t\n"))) {
      sprintf(err_msg, "Error reading gravity coefficient for %s particle_model.", s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
    goma_ctx->particles.model_data[1] = atof(s_ptr2);
    printf("Setting gravity coefficient = %g\n", goma_ctx->particles.model_data[1]);
    break;
  case TRACER_EXPLICIT:
  case TRACER_IMPLICIT:
    s_ptr1 = s_tmp + 15;
    if (!(s_ptr2 = strtok(s_ptr1, " \t\n"))) {
      sprintf(err_msg, "Error reading random walk coefficient for %s particle model.", s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
    goma_ctx->particles.model_data[0] = atof(s_ptr2);
    printf("Setting random walk coefficient = %g\n", goma_ctx->particles.model_data[0]);
    break;
  case SWIMMER_EXPLICIT:
  case SWIMMER_IMPLICIT:
    s_ptr1 = s_tmp + 16;
    if (!(s_ptr2 = strtok(s_ptr1, " \t\n"))) {
      sprintf(err_msg, "Error reading random swimming speed coefficient for %s particle model.",
              s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
    goma_ctx->particles.model_data[0] = atof(s_ptr2);
    printf("Setting random swimming speed coefficient = %g\n", goma_ctx->particles.model_data[0]);
    if (!(s_ptr2 = strtok(NULL, " \t\n"))) {
      sprintf(err_msg, "Error reading random cell orientation coefficient for %s particle model.",
              s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
    goma_ctx->particles.model_data[1] = atof(s_ptr2);
    printf("Setting random cell orientation coefficient = %g\n", goma_ctx->particles.model_data[1]);
    if (!(s_ptr2 = strtok(NULL, " \t\n"))) {
      sprintf(err_msg, "Error reading upswimming speed for %s particle model.", s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
    goma_ctx->particles.model_data[2] = atof(s_ptr2);
    printf("Setting particle upswimming speed = %g\n", goma_ctx->particles.model_data[2]);
    break;
  case CHARGED_TRACER_EXPLICIT:
  case CHARGED_TRACER_IMPLICIT:
    s_ptr1 = s_tmp + 23;
    if (!(s_ptr2 = strtok(s_ptr1, " \t\n"))) {
      sprintf(err_msg, "Error reading random walk coefficient for %s particle model.", s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
    goma_ctx->particles.model_data[0] = atof(s_ptr2);
    printf("Setting random walk coefficient = %g\n", goma_ctx->particles.model_data[0]);
    if (!(s_ptr2 = strtok(NULL, " \t\n"))) {
      sprintf(err_msg, "Error reading charge coefficient for %s particle model.", s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
    goma_ctx->particles.model_data[1] = atof(s_ptr2);
    printf("Setting charge coefficient = %g\n", goma_ctx->particles.model_data[1]);
    break;
  case DIELECTROPHORETIC_TRACER_IMPLICIT:
    s_ptr1 = s_tmp + 33;
    if (!(s_ptr2 = strtok(s_ptr1, " \t\n"))) {
      sprintf(err_msg, "Error reading random walk coefficient for %s particle model.", s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
    goma_ctx->particles.model_data[0] = atof(s_ptr2);
    printf("Setting random walk coefficient = %g\n", goma_ctx->particles.model_data[0]);
    if (!(s_ptr2 = strtok(NULL, " \t\n"))) {
      sprintf(err_msg, "Error reading particle permittivity for %s particle model.", s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
    goma_ctx->particles.model_data[1] = atof(s_ptr2);
    printf("Setting particle permittivity = %g\n", goma_ctx->particles.model_data[1]);
    if (!(s_ptr2 = strtok(NULL, " \t\n"))) {
      sprintf(err_msg, "Error reading medium (fluid) permittivity for %s particle model.", s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
    goma_ctx->particles.model_data[2] = atof(s_ptr2);
    printf("Setting medium (fluid) permittivity = %g\n", goma_ctx->particles.model_data[2]);
    if (!(s_ptr2 = strtok(NULL, " \t\n"))) {
      sprintf(err_msg, "Error reading particle conductivity for %s particle model.", s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
    goma_ctx->particles.model_data[3] = atof(s_ptr2);
    printf("Setting particle conductivity = %g\n", goma_ctx->particles.model_data[3]);
    if (!(s_ptr2 = strtok(NULL, " \t\n"))) {
      sprintf(err_msg, "Error reading medium (fluid) conductivity for %s particle model.", s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
    goma_ctx->particles.model_data[4] = atof(s_ptr2);
    printf("Setting medium (fluid) conductivity = %g\n", goma_ctx->particles.model_data[4]);
    if (!(s_ptr2 = strtok(NULL, " \t\n"))) {
      sprintf(err_msg, "Error reading AC angular frequency for %s particle model.", s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
    goma_ctx->particles.model_data[5] = atof(s_ptr2);
    printf("Setting AC angular frequency = %g\n", goma_ctx->particles.model_data[5]);
    if (!(s_ptr2 = strtok(NULL, " \t\n"))) {
      sprintf(err_msg, "Error reading Volt conversion factor for %s particle model.", s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
    goma_ctx->particles.model_data[6] = atof(s_ptr2);
    printf("Setting Volt conversion factor = %g\n", goma_ctx->particles.model_data[6]);
    break;
  default:
    GOMA_EH(GOMA_ERROR, "How'd you get here?");
  }

  /* Get maximum number of particle time steps if we're running after
   * a steady-state solution... */
  if (TimeIntegration == STEADY) {
    if (look_for_optional(ifp, "Time steps", input, '=') == 1) {
      if (fscanf(ifp, "%d", &goma_ctx->particles.max_time_steps) != 1)
        GOMA_EH(GOMA_ERROR, "Error reading Time steps card for post-steady problem.");
      printf("Setting Particle_Max_Time_Steps = %d\n", goma_ctx->particles.max_time_steps);
    } else
      goma_ctx->particles.max_time_steps = 0;
  } else if (look_for_optional(ifp, "Time steps", input, '=') == 1)
    printf("CAUTION: ignoring Time steps card for transient problem.");

  /* How many particles would you like?  For now, they are always
   * introduced unformly throughout the domain. */
  look_for(ifp, "Number of particles", input, '=');
  if (fscanf(ifp, "%d", &goma_ctx->particles.number) != 1)
    GOMA_EH(GOMA_ERROR, "Error reading Number of particles card.");
  printf("Setting Particle_Number = %d\n", goma_ctx->particles.number);

  if (look_for_optional(ifp, "Restart file", input, '=') == 1) {
    char file_name_temp[MAX_PARTICLE_FILENAME_LENGTH];
    if (!fgets(file_name_temp, MAX_PARTICLE_FILENAME_LENGTH, ifp))
      GOMA_EH(GOMA_ERROR, "Error reading Restart file card.");
    strip(file_name_temp);
    strncpy(goma_ctx->particles.restart_filename, file_name_temp, MAX_PARTICLE_FILENAME_LENGTH);
    printf("Restart file = %s\n", goma_ctx->particles.restart_filename);
  } else
    sprintf(goma_ctx->particles.restart_filename, "<not active>");

  /* When to output particle information.  There are two ways to
   * specify this, and they are incompatible (i.e., only one can be
   * specified).  Either you can output particles every so many Goma
   * time steps, or you can have particles output every so often
   * time-wise (e.g., every 0.1 seconds), whether within a Goma time
   * step or containing many Goma time steps.*/
  if (look_for_optional(ifp, "Output stride", input, '=') == 1) {
    if (fscanf(ifp, "%d", &goma_ctx->particles.output_stride) != 1)
      GOMA_EH(GOMA_ERROR, "Error reading Output stride card.");
    printf("Setting Particle_Output_Stride = %d\n", goma_ctx->particles.output_stride);
  } else
    goma_ctx->particles.output_stride = -1;

  if (look_for_optional(ifp, "Output time step", input, '=') == 1) {
    if (fscanf(ifp, "%lf", &goma_ctx->particles.output_time_step) != 1)
      GOMA_EH(GOMA_ERROR, "Error reading Output time step card.");
    printf("Setting Particle_Output_Time_Step = %g\n", goma_ctx->particles.output_time_step);
    if (goma_ctx->particles.output_stride != -1)
      GOMA_EH(GOMA_ERROR, "You cannot specify both Output stride and Output time step.");
  } else
    goma_ctx->particles.output_time_step = 0.0;

  if (goma_ctx->particles.output_stride == -1 && goma_ctx->particles.output_time_step == 0.0)
    GOMA_EH(GOMA_ERROR, "One of the Output stride or Output time step cards must be present.");

  if (look_for_optional(ifp, "Output format", input, '=') == 1) {
    if (!fgets(s_tmp, SLEN - 1, ifp))
      GOMA_EH(GOMA_ERROR, "Error reading Output format card.");
    strip(s_tmp); /* s_tmp still has '\n' at end -- don't care. */
    if (!strncmp("TECPLOT", s_tmp, 7))
      goma_ctx->particles.output_format = TECPLOT;
    else if (!strncmp("FLAT_TEXT", s_tmp, 9))
      goma_ctx->particles.output_format = FLAT_TEXT;
    else {
      sprintf(s_tmp_save, "Unknown Output format: %s\n", s_tmp);
      GOMA_EH(GOMA_ERROR, s_tmp_save);
    }
  } else
    goma_ctx->particles.output_format = FLAT_TEXT;

  if (look_for_optional(ifp, "Particle density", input, '=') == 1) {
    if (fscanf(ifp, "%lf", &goma_ctx->particles.density) != 1)
      GOMA_EH(GOMA_ERROR, "Error reading Particle density card.");
  } else
    goma_ctx->particles.density = 1.0;
  printf("Setting Particle_Density = %g\n", goma_ctx->particles.density);

  if (look_for_optional(ifp, "Particle radius", input, '=') == 1) {
    if (fscanf(ifp, "%lf", &goma_ctx->particles.radius) != 1)
      GOMA_EH(GOMA_ERROR, "Error reading Particle radius card.");
  } else
    goma_ctx->particles.radius = 1.0;
  printf("Setting Particle_Radius = %g\n", goma_ctx->particles.radius);
  if (goma_ctx->particles.radius <= 0.0)
    GOMA_EH(
        GOMA_ERROR,
        "Particle radius must be > 0.0.  Non-inertial tracer particles will ignore this setting.");

  if (look_for_optional(ifp, "Particle ratio", input, '=') == 1) {
    if (fscanf(ifp, "%lf", &goma_ctx->particles.ratio) != 1)
      GOMA_EH(GOMA_ERROR, "Error reading Particle ratio card.");
  } else
    goma_ctx->particles.ratio = 1.0;
  printf("Setting Particle_Ratio = %g\n", goma_ctx->particles.ratio);

  goma_ctx->particles.show_debug_info = 0;
  if (look_for_optional(ifp, "Show particle debug info", input, '=') == 1) {
    if (fscanf(ifp, "%s", s_tmp) != 1)
      GOMA_EH(GOMA_ERROR, "Need yes or no for Show particle debug info card.");
    if (!strncmp(s_tmp, "YES", 3)) {
      goma_ctx->particles.show_debug_info = 1;
      printf("Showing particle debug info\n");
    }
  }
  if (!goma_ctx->particles.show_debug_info)
    printf("Not showing particle debug info\n");

  /* Initialization for particle creation and move domains. */
  for (i = 0; i < MAX_DOMAIN_REAL_VALUES; i++) {
    goma_ctx->particles.creation_domain_reals[i] = 1.0e+10;
    goma_ctx->particles.move_domain_reals[i] = 1.0e+10;
  }
  goma_ctx->particles.creation_domain = UNRESTRICTED;
  goma_ctx->particles.move_domain = UNRESTRICTED;
  sprintf(goma_ctx->particles.creation_domain_filename, "<not active>");
  sprintf(goma_ctx->particles.creation_domain_name, "<not active>");
  sprintf(goma_ctx->particles.move_domain_filename, "<not active>");
  sprintf(goma_ctx->particles.move_domain_name, "<not active>");

  /* Set optional creation domain. */
  if (look_for_optional(ifp, "Particle creation domain", input, '=') == 1) {
    if (fscanf(ifp, "%s", s_tmp) != 1)
      GOMA_EH(GOMA_ERROR, "Error reading Particle creation domain card.");
    if (!strncmp(s_tmp, "BRICK", 5)) {
      goma_ctx->particles.creation_domain = BRICK;
      if (fscanf(ifp, "%lf %lf %lf %lf %lf %lf\n", &goma_ctx->particles.creation_domain_reals[0],
                 &goma_ctx->particles.creation_domain_reals[1],
                 &goma_ctx->particles.creation_domain_reals[2],
                 &goma_ctx->particles.creation_domain_reals[3],
                 &goma_ctx->particles.creation_domain_reals[4],
                 &goma_ctx->particles.creation_domain_reals[5]) != 6)
        GOMA_EH(GOMA_ERROR, "Error reading min/max values on Particle creation domain card.");
      printf("Using BRICK creation domain with x in [%g,%g], y in [%g,%g], z in [%g,%g]\n",
             goma_ctx->particles.creation_domain_reals[0],
             goma_ctx->particles.creation_domain_reals[1],
             goma_ctx->particles.creation_domain_reals[2],
             goma_ctx->particles.creation_domain_reals[3],
             goma_ctx->particles.creation_domain_reals[4],
             goma_ctx->particles.creation_domain_reals[5]);
    } else if (!strncmp(s_tmp, "ACIS", 4)) {
      GOMA_EH(GOMA_ERROR, "CGM not supported, ACIS");
    } else {
      sprintf(err_msg, "Error reading Particle creation domain card, unknown domain type: %s.",
              s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
  } else
    printf("Using unrestricted Particle_Creation_Domain\n");

  /* Set optional move domain. */
  if (look_for_optional(ifp, "Particle move domain", input, '=') == 1) {
    if (fscanf(ifp, "%s", s_tmp) != 1)
      GOMA_EH(GOMA_ERROR, "Error reading Particle move domain card.");
    if (!strncmp(s_tmp, "BRICK", 5)) {
      goma_ctx->particles.move_domain = BRICK;
      if (fscanf(
              ifp, "%lf %lf %lf %lf %lf %lf\n", &goma_ctx->particles.move_domain_reals[0],
              &goma_ctx->particles.move_domain_reals[1], &goma_ctx->particles.move_domain_reals[2],
              &goma_ctx->particles.move_domain_reals[3], &goma_ctx->particles.move_domain_reals[4],
              &goma_ctx->particles.move_domain_reals[5]) != 6)
        GOMA_EH(GOMA_ERROR, "Error reading min/max values on Particle move domain card.");
      printf("Using BRICK move domain with x in [%g,%g], y in [%g,%g], z in [%g,%g]\n",
             goma_ctx->particles.move_domain_reals[0], goma_ctx->particles.move_domain_reals[1],
             goma_ctx->particles.move_domain_reals[2], goma_ctx->particles.move_domain_reals[3],
             goma_ctx->particles.move_domain_reals[4], goma_ctx->particles.move_domain_reals[5]);
    } else if (!strncmp(s_tmp, "ACIS", 4)) {
      GOMA_EH(GOMA_ERROR, "Need the CGM library to use ACIS for Particle creation domain card.");
    } else {
      sprintf(err_msg, "Error reading Particle move domain card, unknown domain type: %s.", s_tmp);
      GOMA_EH(GOMA_ERROR, err_msg);
    }
  } else
    printf("Using unrestricted Particle_Move_Domain\n");

  /* This is used for restarts, too, and ignores output file type.  If
   * we're a steady-state Goma calculation, then this applies to the
   * particle timesteps taken afterwards.  If this is a transient Goma
   * calculation, then this is a multiple of Goma steps.  In
   * particular, if particles are output on a time strided basis, that
   * will be ignored here. */
  if (look_for_optional(ifp, "Full output stride", input, '=') == 1) {
    char file_name_temp[MAX_PARTICLE_FILENAME_LENGTH];
    if (fscanf(ifp, "%d", &goma_ctx->particles.full_output_stride) != 1)
      GOMA_EH(GOMA_ERROR, "Problem reading Full output stride card.");
    if (!fgets(file_name_temp, MAX_PARTICLE_FILENAME_LENGTH, ifp))
      GOMA_EH(GOMA_ERROR, "Problem reading Full output stride card.");
    strip(file_name_temp);
    strncpy(goma_ctx->particles.full_output_filename, file_name_temp, MAX_PARTICLE_FILENAME_LENGTH);
    printf("Setting full output every %d steps to file %s\n",
           goma_ctx->particles.full_output_stride, goma_ctx->particles.full_output_filename);
  } else {
    goma_ctx->particles.full_output_stride = 0;
    sprintf(goma_ctx->particles.full_output_filename, "<not active>");
  }

  /* General description for number of particles to track, and how
   * many of them.  Can specify multiple selections, and the particles
   * will be separate samples.  Subject to total number of particles,
   * of course... */
  look_for(ifp, "Number of sample types", input, '=');
  if (fscanf(ifp, "%d", &goma_ctx->particles.number_sample_types) != 1)
    GOMA_EH(GOMA_ERROR, "Problem reading Number of sample types card.");
  printf("Setting Particle_Number_Sample_Types = %d\n", goma_ctx->particles.number_sample_types);

  if (goma_ctx->particles.number_sample_types) {
    goma_ctx->particles.number_samples_existing =
        (int *)array_alloc(1, goma_ctx->particles.number_sample_types, sizeof(int));
    goma_ctx->particles.number_samples =
        (int *)array_alloc(1, goma_ctx->particles.number_sample_types, sizeof(int));
    goma_ctx->particles.number_output_variables =
        (int *)array_alloc(1, goma_ctx->particles.number_sample_types, sizeof(int));
    goma_ctx->particles.output_variables = (particle_variable_s **)array_alloc(
        1, goma_ctx->particles.number_sample_types, sizeof(particle_variable_s *));
    goma_ctx->particles.filename_template = (particle_filename_s *)array_alloc(
        1, goma_ctx->particles.number_sample_types, sizeof(particle_filename_s));
  } else {
    goma_ctx->particles.number_samples_existing = NULL;
    goma_ctx->particles.number_samples = NULL;
    goma_ctx->particles.number_output_variables = NULL;
    goma_ctx->particles.output_variables = NULL;
    goma_ctx->particles.filename_template = NULL;
  }

  for (i = 0; i < goma_ctx->particles.number_sample_types; i++) {
    look_for(ifp, "Sample", input, '=');
    if (fscanf(ifp, "%d", &goma_ctx->particles.number_samples[i]) != 1)
      GOMA_EH(GOMA_ERROR, "Error reading number of particle samples.");
    if (goma_ctx->particles.number_samples[i] > 0)
      printf("Sample %d contains %d particles", i, goma_ctx->particles.number_samples[i]);
    else
      GOMA_EH(GOMA_ERROR, "Number of particle samples must be > 0.");

    if (!fgets(s_tmp, SLEN - 1, ifp))
      GOMA_EH(GOMA_ERROR, "Error reading sample variables and filename.");
    strncpy(s_tmp_save, s_tmp, SLEN);
    goma_ctx->particles.number_output_variables[i] = 0;

    s_ptr1 = strtok(s_tmp, " \t\n");
    while ((s_ptr1 = strtok(NULL, " \t\n")))
      goma_ctx->particles.number_output_variables[i]++;

    s_ptr1 = strtok(s_tmp_save, " \t\n");
    if (goma_ctx->particles.number_output_variables[i] > 0) {
      printf(" tracking variable(s)");
      goma_ctx->particles.output_variables[i] = (particle_variable_s *)calloc(
          goma_ctx->particles.number_output_variables[i], sizeof(particle_variable_s));
      /*
        Particle_Output_Variables[i] = (char **)array_alloc(1, Particle_Number_Output_Variables[i],
        sizeof(char *));
      */
      for (j = 0; j < goma_ctx->particles.number_output_variables[i]; j++) {
        if (strlen(s_ptr1) + 1 > MAX_PARTICLE_OUTPUT_VARIABLE_LENGTH)
          GOMA_EH(GOMA_ERROR, "Increase MAX_PARTICLE_OUTPUT_VARIABLE_LENGTH");
        strncpy(goma_ctx->particles.output_variables[i][j], s_ptr1, strlen(s_ptr1) + 1);
        printf(" %s", goma_ctx->particles.output_variables[i][j]);
        s_ptr1 = strtok(NULL, " \t\n");
      }
    } else {
      goma_ctx->particles.output_variables[i] = NULL;
    }

    if (strlen(s_ptr1) + 1 > MAX_PARTICLE_FILENAME_LENGTH)
      GOMA_EH(GOMA_ERROR, "Increase MAX_PARTICLE_FILENAME_LENGTH");
    strncpy(goma_ctx->particles.filename_template[i], s_ptr1, strlen(s_ptr1) + 1);
    goma_ctx->particles.filename_template[i][strlen(s_ptr1)] = 0;
    printf(" to file %s.\n", goma_ctx->particles.filename_template[i]);
  }

  iread = look_for_optional(ifp, "START OF PBC", input, '=');
  if (iread == 1) {
    goma_ctx->particles.num_pbcs = count_list(ifp, "PBC", input, '=', "END OF PBC");
    printf("Found %d PBC's\n", goma_ctx->particles.num_pbcs);
    goma_ctx->particles.pbcs = (PBC_t *)calloc(goma_ctx->particles.num_pbcs, sizeof(PBC_t));
    for (i = 0; i < goma_ctx->particles.num_pbcs; i++) {
      PBC = &goma_ctx->particles.pbcs[i];
      look_for(ifp, "PBC", input, '=');
      if (fscanf(ifp, "%80s", input) != 1)
        GOMA_EH(GOMA_ERROR, "Error reading PBC.");
      stringup(input);

      /* I know it is stupid to do this for an unnecssary token,
       * but it makes the input look better... */
      if (fscanf(ifp, "%s", s_tmp) != 1)
        GOMA_EH(GOMA_ERROR, "Missing SS token on PBC card.");
      if (strncmp(s_tmp, "SS", 2))
        GOMA_EH(GOMA_ERROR, "Missing SS token on PBC card.");

      if (fscanf(ifp, "%d", &(PBC->SS_id)) != 1)
        GOMA_EH(GOMA_ERROR, "Error reading SS_id on PBC card.");

      if (!strncmp(input, "OUTFLOW", 7)) {
        PBC->type = PBC_OUTFLOW;
        puts("Found an OUTFLOW PBC.");
      } else if (!strncmp(input, "SOURCE", 6)) {
        PBC->type = PBC_SOURCE;
        if (fscanf(ifp, "%lf", &(PBC->real_data[0])) != 1)
          GOMA_EH(GOMA_ERROR, "Error reading SS specification and float on PBC = SOURCE card.");
        printf("Found a SOURCE PBC with source = %g.\n", PBC->real_data[0]);
      } else if (!strncmp(input, "TARGET", 6)) {
        PBC->type = PBC_TARGET;
        if (fscanf(ifp, "%s", s_tmp) != 1)
          GOMA_EH(GOMA_ERROR, "Missing filename on PBC TARGET card.");
        if (strlen(s_tmp) + 1 > MAX_PBC_STRING_DATA_LENGTH)
          GOMA_EH(GOMA_ERROR, "Increase MAX_PBC_STRING_DATA_LENGTH");
        strncpy(PBC->string_data, s_tmp, strlen(s_tmp) + 1);
        printf("Found a TARGET PBC with filename = '%s'.\n", PBC->string_data);
      } else if (!strncmp(input, "FREESTREAM_SOURCE", 17)) {
        PBC->type = PBC_FREESTREAM_SOURCE;
        if (fscanf(ifp, "%lf", &(PBC->real_data[0])) != 1)
          GOMA_EH(GOMA_ERROR,
                  "Error reading SS specification and float on PBC = FREESTREAM_SOURCE card.");
        printf("Found a FREESTREAM_SOURCE PBC with source = %g.\n", PBC->real_data[0]);
      } else if (!strncmp(input, "IMPERMEABLE", 11)) {
        PBC->type = PBC_IMPERMEABLE;
        if (fscanf(ifp, "%lf", &(PBC->real_data[0])) != 1)
          GOMA_EH(GOMA_ERROR, "Error reading distance factor on PBC = IMPERMEABLE card.");
        if (!fgets(s_tmp, SLEN - 1, ifp))
          GOMA_EH(GOMA_ERROR, "Error reading geometry entity name on PBC = IMPERMEABLE card.");
        strip(s_tmp);
        if (s_tmp[strlen(s_tmp) - 1] == '\n')
          s_tmp[strlen(s_tmp) - 1] = 0;
        if (strlen(s_tmp) + 1 > MAX_PBC_STRING_DATA_LENGTH)
          GOMA_EH(GOMA_ERROR, "Increase MAX_PBC_STRING_DATA_LENGTH");
        strncpy(PBC->string_data, s_tmp, strlen(s_tmp) + 1);
        printf("Found an IMPERMEABLE PBC with distance factor = %g to entity '%s'.\n",
               PBC->real_data[0], PBC->string_data);
      } else
        GOMA_EH(GOMA_ERROR, "Error reading PBC type.");
    }
  } else {
    goma_ctx->particles.num_pbcs = 0;
    goma_ctx->particles.pbcs = NULL;
  }
}
