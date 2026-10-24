/************************************************************************ *
* Goma - Multiphysics finite element software                             *
* Sandia National Laboratories                                            *
*                                                                         *
* Copyright (c) Goma Developers, National Technology & Engineering        *
*               Solutions of Sandia, LLC (NTESS)                          *
*                                                                         *
* Under the terms of Contract DE-NA0003525, the U.S. Government retains   *
* certain rights in this software.                                        *
*                                                                         *
* This software is distributed under the GNU General Public License.      *
* See LICENSE file.                                                       *
\************************************************************************/

/*
 * Debugging aid for library mode (GOMA_CHECK_GLOBAL_RESET)
 *
 * Snapshots the writable data segments (.data/.bss) of the executable and of
 * any loaded libgoma shared library, and later reports the byte ranges that
 * differ from the snapshot.  Taking the snapshot in goma_library_initialize and
 * comparing after goma_destroy finds global and static variables that are not
 * restored to their load time values.
 *
 * Differences are printed as "<object> <offset> <length>", offsets are virtual
 * addresses within the object, use scripts/resolve_reset_check.py to map them
 * to symbol names.
 */

#ifdef GOMA_CHECK_GLOBAL_RESET

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <elf.h>
#include <link.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "goma_reset_check.h"

#define MAX_SEGMENTS 16

struct snapshot_segment {
  char object[1024];
  uintptr_t base;  /* load address of the object */
  uintptr_t vaddr; /* segment virtual address within the object */
  size_t size;
  unsigned char *copy;
};

static struct snapshot_segment segments[MAX_SEGMENTS];
static int num_segments = 0;

static int is_checked_object(const char *name) {
  /* empty name is the main executable */
  return name[0] == '\0' || strstr(name, "libgoma") != NULL;
}

static int add_segments(struct dl_phdr_info *info, size_t size, void *data) {
  (void)size;
  (void)data;
  if (!is_checked_object(info->dlpi_name)) {
    return 0;
  }
  for (int i = 0; i < info->dlpi_phnum; i++) {
    const ElfW(Phdr) *ph = &info->dlpi_phdr[i];
    if (ph->p_type != PT_LOAD || !(ph->p_flags & PF_W) || num_segments >= MAX_SEGMENTS) {
      continue;
    }
    struct snapshot_segment *seg = &segments[num_segments++];
    snprintf(seg->object, sizeof(seg->object), "%s",
             info->dlpi_name[0] == '\0' ? "/proc/self/exe" : info->dlpi_name);
    seg->base = info->dlpi_addr;
    seg->vaddr = ph->p_vaddr;
    seg->size = ph->p_memsz;
    seg->copy = malloc(seg->size);
    memcpy(seg->copy, (void *)(seg->base + seg->vaddr), seg->size);
  }
  return 0;
}

void goma_reset_check_snapshot(void) {
  goma_reset_check_free();
  dl_iterate_phdr(add_segments, NULL);
}

int goma_reset_check_compare(FILE *out) {
  int num_ranges = 0;
  for (int s = 0; s < num_segments; s++) {
    struct snapshot_segment *seg = &segments[s];
    const unsigned char *now = (const unsigned char *)(seg->base + seg->vaddr);
    size_t i = 0;
    while (i < seg->size) {
      if (now[i] == seg->copy[i]) {
        i++;
        continue;
      }
      size_t start = i;
      while (i < seg->size && now[i] != seg->copy[i]) {
        i++;
      }
      fprintf(out, "%s 0x%zx %zu\n", seg->object, (size_t)(seg->vaddr + start), i - start);
      num_ranges++;
    }
  }
  return num_ranges;
}

void goma_reset_check_free(void) {
  for (int s = 0; s < num_segments; s++) {
    free(segments[s].copy);
  }
  num_segments = 0;
}

#endif /* GOMA_CHECK_GLOBAL_RESET */
