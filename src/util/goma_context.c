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

#include "util/goma_context.h"

#include <limits.h>
#include <string.h>

void goma_context_init(GomaContext *ctx) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->num_proc = 1;
  ctx->proc_id = 0;
  ctx->dim = -1;
  ctx->request = NULL;
  ctx->status = NULL;
  ctx->num_requests = 6;
  ctx->unlimited_output = 1;
  ctx->exo_time_plane = INT_MAX;
  ctx->decompose_flag = 1;
  strncpy(ctx->front_scratch_directory, FRONT_SCRATCH_DIRECTORY, MAX_FNL - 1);
}

/* Static storage with the same defaults goma_context_init() applies. */
static GomaContext goma_context_storage = {.num_proc = 1,
                                           .dim = -1,
                                           .num_requests = 6,
                                           .unlimited_output = 1,
                                           .exo_time_plane = INT_MAX,
                                           .decompose_flag = 1,
                                           .front_scratch_directory = FRONT_SCRATCH_DIRECTORY};

GomaContext *goma_ctx = &goma_context_storage;
