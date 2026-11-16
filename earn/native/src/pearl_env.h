// One reader for the cores' numeric environment switches (CUDA, HIP and SYCL hosts).

#ifndef PEARL_ENV_H
#define PEARL_ENV_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// The unsigned 32-bit value of environment variable `name`, into *out. Returns false,
// leaving *out alone, when it is unset or empty; a value that is not all digits or is
// over 32 bits is reported on stderr and ignored the same way.
//
// Parsed by hand, not with strtoul. Under glibc 2.38's headers strtoul becomes
// __isoc23_strtoul, so an addon built against them needs glibc 2.38 and will not load
// on Ubuntu 22.04 or Debian 12 (v0.6.7 shipped that way). The HIP core still builds on
// Ubuntu 24.04; native-core.yml checks every Linux core's glibc floor.
inline bool pearl_env_u32(const char *name, uint32_t *out) {
  const char *v = getenv(name);
  if (!v || !*v) return false;
  uint64_t x = 0;
  for (const char *p = v; *p; p++) {
    if (*p < '0' || *p > '9' || (x = x * 10u + (uint64_t)(*p - '0')) > 0xFFFFFFFFull) {
      fprintf(stderr, "[pearl] %s=%s ignored: not a number\n", name, v);
      return false;
    }
  }
  *out = (uint32_t)x;
  return true;
}

#endif  // PEARL_ENV_H
