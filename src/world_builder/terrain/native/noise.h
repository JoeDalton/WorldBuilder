/* What the noise library's own files share (noise.c, scatter.c): the lattice
 * hash and the fbm sum at one point. Internal: nothing here is exported
 * (the library builds with -fvisibility=hidden). Plain C99. */
#ifndef WB_NOISE_H
#define WB_NOISE_H

#include <stdint.h>

/* The splitmix64 finaliser: spreads a 64-bit value over all 64 bits. */
static inline uint64_t wb_mix64(uint64_t h) {
  h ^= h >> 30;
  h *= 0xBF58476D1CE4E5B9ULL;
  h ^= h >> 27;
  h *= 0x94D049BB133111EBULL;
  h ^= h >> 31;
  return h;
}

/* noise._hash: lattice coordinates and seed into 64 bits. */
static inline uint64_t wb_lattice_hash(int64_t i, int64_t j, int64_t k,
                                       uint64_t seed) {
  uint64_t h = ((uint64_t)i * 0x9E3779B185EBCA87ULL) ^
               ((uint64_t)j * 0xC2B2AE3D27D4EB4FULL);
  h ^= (uint64_t)k * 0x165667B19E3779F9ULL;
  h ^= seed;
  return wb_mix64(h);
}

/* noise.fbm's weighted sum of octaves at one point, before its
 * normalisation (wb_fbm per point). frames: (octaves, 12); seeds: mixed. */
double wb_fbm_point(const double *p, int32_t octaves, const double *frames,
                    const uint64_t *seeds, const double *weights);

#endif
