/* Native noise for world_builder.terrain.noise and clouds.noise (issue #20).
 *
 * The same 3D simplex noise as noise.py (hash, gradients, kernel, scale),
 * summed over octaves per point, so a chunk's noise is one call per field
 * instead of ~150 numpy operations per octave. Python still chooses the
 * octaves: it passes each one's frame (rotation scaled by the frequency, and
 * offset), mixed seed and weight. A single octave gives the same bits as
 * noise.simplex3; sums of octaves differ from numpy in the last bits only,
 * because numpy rotates the points with BLAS.
 *
 * Plain C, no Python or Panda headers: loaded with ctypes. Build without
 * floating-point contraction (-ffp-contract=off), so every platform rounds the
 * same way and gives the same terrain. */
#include <math.h>
#include <stdint.h>

#include "swell.h"

#if defined(_WIN32)
#define WB_API __declspec(dllexport)
#else
#define WB_API __attribute__((visibility("default")))
#endif

/* Bumped when a signature or a result changes; noise.py checks it. */
#define WB_NOISE_ABI 5

static const double F3 = 1.0 / 3.0;
static const double G3 = 1.0 / 6.0;
static const double R2 = 0.5;
static const double SCALE = 76.0;

/* noise._GRADIENTS: the 12 cube-edge gradients, padded to 16. */
static const double GRADIENTS[16][3] = {
    {1, 1, 0},  {-1, 1, 0},  {1, -1, 0}, {-1, -1, 0}, {1, 0, 1},  {-1, 0, 1},
    {1, 0, -1}, {-1, 0, -1}, {0, 1, 1},  {0, -1, 1},  {0, 1, -1}, {0, -1, -1},
    {1, 1, 0},  {-1, 1, 0},  {0, -1, 1}, {0, -1, -1},
};

/* noise._hash: lattice coordinates and seed into 64 bits (splitmix64). */
static uint64_t lattice_hash(int64_t i, int64_t j, int64_t k, uint64_t seed) {
  uint64_t h = ((uint64_t)i * 0x9E3779B185EBCA87ULL) ^
               ((uint64_t)j * 0xC2B2AE3D27D4EB4FULL);
  h ^= (uint64_t)k * 0x165667B19E3779F9ULL;
  h ^= seed;
  h ^= h >> 30;
  h *= 0xBF58476D1CE4E5B9ULL;
  h ^= h >> 27;
  h *= 0x94D049BB133111EBULL;
  h ^= h >> 31;
  return h;
}

/* noise._corner */
static double corner(double x, double y, double z, int64_t i, int64_t j,
                     int64_t k, uint64_t seed) {
  double t = R2 - (x * x + y * y + z * z);
  const double *g;
  if (t < 0.0) t = 0.0;
  t *= t;
  t *= t;
  g = GRADIENTS[lattice_hash(i, j, k, seed) >> 60];
  return t * (g[0] * x + g[1] * y + g[2] * z);
}

/* noise.simplex3 at one point. */
static double simplex(double x, double y, double z, uint64_t seed) {
  double skew = (x + y + z) * F3;
  double fi = floor(x + skew), fj = floor(y + skew), fk = floor(z + skew);
  double unskew = (fi + fj + fk) * G3;
  double x0 = x - fi + unskew, y0 = y - fj + unskew, z0 = z - fk + unskew;
  int64_t i = (int64_t)fi, j = (int64_t)fj, k = (int64_t)fk;
  /* Rank of each offset among the three picks the simplex. */
  int xy = x0 >= y0, xz = x0 >= z0, yz = y0 >= z0;
  int rank_x = xy + xz, rank_y = !xy + yz, rank_z = !xz + !yz;
  int i1 = rank_x >= 2, j1 = rank_y >= 2, k1 = rank_z >= 2;
  int i2 = rank_x >= 1, j2 = rank_y >= 1, k2 = rank_z >= 1;
  double n = corner(x0, y0, z0, i, j, k, seed);
  n += corner(x0 - i1 + G3, y0 - j1 + G3, z0 - k1 + G3, i + i1, j + j1, k + k1,
              seed);
  n += corner(x0 - i2 + 2.0 * G3, y0 - j2 + 2.0 * G3, z0 - k2 + 2.0 * G3,
              i + i2, j + j2, k + k2, seed);
  n += corner(x0 - (1.0 - 3.0 * G3), y0 - (1.0 - 3.0 * G3),
              z0 - (1.0 - 3.0 * G3), i + 1, j + 1, k + 1, seed);
  return n * SCALE;
}

/* One octave at point p: the point through its frame (noise._octave), then
 * the noise. frame: 3x3 matrix (rotation.T * frequency, row-major), then the
 * offset. */
static double octave(const double *p, const double *frame, uint64_t seed) {
  const double *m = frame, *o = frame + 9;
  double x = p[0] * m[0] + p[1] * m[3] + p[2] * m[6] + o[0];
  double y = p[0] * m[1] + p[1] * m[4] + p[2] * m[7] + o[1];
  double z = p[0] * m[2] + p[1] * m[5] + p[2] * m[8] + o[2];
  return simplex(x, y, z, seed);
}

WB_API int wb_noise_abi(void) { return WB_NOISE_ABI; }

/* noise.simplex3. points: (n, 3); seed: already mixed (noise._mix_seed). */
WB_API void wb_simplex3(const double *points, int64_t n, uint64_t seed,
                        double *out) {
  int64_t a;
  for (a = 0; a < n; a++)
    out[a] = simplex(points[3 * a], points[3 * a + 1], points[3 * a + 2], seed);
}

/* The weighted sum of octaves, as noise.fbm before its normalisation:
 * out = sum of weights[o] * octave o. frames: (octaves, 12); seeds: mixed. */
WB_API void wb_fbm(const double *points, int64_t n, int32_t octaves,
                   const double *frames, const uint64_t *seeds,
                   const double *weights, double *out) {
  int64_t a;
  int32_t o;
  for (a = 0; a < n; a++) {
    const double *p = points + 3 * a;
    double total = 0.0;
    for (o = 0; o < octaves; o++)
      total += weights[o] * octave(p, frames + 12 * o, seeds[o]);
    out[a] = total;
  }
}

/* noise.ridged before its normalisation, over the octaves it evaluates
 * (those of zero band-limit weight left out by the caller: they change
 * nothing). Octave 0 sets the signal; octave o > 0 is fed by the previous
 * signal and adds weights[o] (its amplitude times band-limit weight) times
 * its signal. weights[0] is unused. */
WB_API void wb_ridged(const double *points, int64_t n, int32_t octaves,
                      const double *frames, const uint64_t *seeds,
                      const double *weights, double offset, double gain,
                      double *out) {
  int64_t a;
  int32_t o;
  for (a = 0; a < n; a++) {
    const double *p = points + 3 * a;
    double signal = offset - fabs(octave(p, frames, seeds[0]));
    double total;
    signal *= signal;
    total = signal;
    for (o = 1; o < octaves; o++) {
      double feed = signal * gain;
      if (feed < 0.0) feed = 0.0;
      if (feed > 1.0) feed = 1.0;
      signal = offset - fabs(octave(p, frames + 12 * o, seeds[o]));
      signal *= signal;
      signal *= feed;
      total += weights[o] * signal;
    }
    out[a] = total;
  }
}

/* The clouds' noise (clouds/noise.py): one tileable octave stored as a
 * (size, size, size) float32 volume indexed [z][y][x], sampled as the GPU
 * samples a repeat-wrapped linear 3D texture (sample_volume), summed over
 * octaves as fbm does. The same operations in the same order as numpy:
 * corner differences in float32, the blends in float64. */
static double sample_volume(const float *v, int64_t size, double x, double y,
                            double z) {
  double bx = floor(x - 0.5), by = floor(y - 0.5), bz = floor(z - 0.5);
  double tx = (x - 0.5) - bx, ty = (y - 0.5) - by, tz = (z - 0.5) - bz;
  /* Python's modulo, never negative: a mask for a power-of-two size (two's
   * complement), which spares six divisions a sample. */
  int64_t x0, y0, z0, x1, y1, z1;
  if ((size & (size - 1)) == 0) {
    int64_t mask = size - 1;
    x0 = (int64_t)bx & mask, y0 = (int64_t)by & mask, z0 = (int64_t)bz & mask;
    x1 = (x0 + 1) & mask, y1 = (y0 + 1) & mask, z1 = (z0 + 1) & mask;
  } else {
    x0 = ((int64_t)bx % size + size) % size;
    y0 = ((int64_t)by % size + size) % size;
    z0 = ((int64_t)bz % size + size) % size;
    x1 = (x0 + 1) % size, y1 = (y0 + 1) % size, z1 = (z0 + 1) % size;
  }
  int64_t row = size, page = size * size;
  float c000 = v[z0 * page + y0 * row + x0], c100 = v[z0 * page + y0 * row + x1];
  float c010 = v[z0 * page + y1 * row + x0], c110 = v[z0 * page + y1 * row + x1];
  float c001 = v[z1 * page + y0 * row + x0], c101 = v[z1 * page + y0 * row + x1];
  float c011 = v[z1 * page + y1 * row + x0], c111 = v[z1 * page + y1 * row + x1];
  double c00 = (double)c000 + (double)(float)(c100 - c000) * tx;
  double c01 = (double)c010 + (double)(float)(c110 - c010) * tx;
  double c10 = (double)c001 + (double)(float)(c101 - c001) * tx;
  double c11 = (double)c011 + (double)(float)(c111 - c011) * tx;
  double c0 = c00 + (c01 - c00) * ty;
  double c1 = c10 + (c11 - c10) * ty;
  return c0 + (c1 - c0) * tz;
}

/* clouds.noise.fbm: octave k samples the volume at coords * scales[k] +
 * offsets[k] and adds weights[k] times it; octaves from `evaluated` on add
 * weights[k] * octave_mean instead. coords: (n, 3). */
WB_API void wb_volume_fbm(const float *volume, int64_t size,
                          const double *coords, int64_t n, int32_t octaves,
                          int32_t evaluated, const double *scales,
                          const double *weights, const double *offsets,
                          double octave_mean, double *out) {
  int64_t a;
  int32_t k;
  for (a = 0; a < n; a++) {
    const double *c = coords + 3 * a;
    double total = 0.0;
    for (k = 0; k < octaves; k++) {
      if (k < evaluated) {
        const double *o = offsets + 3 * k;
        double s = scales[k];
        total += sample_volume(volume, size, c[0] * s + o[0], c[1] * s + o[1],
                               c[2] * s + o[2]) *
                 weights[k];
      } else {
        total += octave_mean * weights[k];
      }
    }
    out[a] = total;
  }
}

/* The sea's swell (swell.h): the water's height at n sea-level points over
 * seabed this deep, each for its own resolution (max_freqs). */
WB_API void wb_swell_heights(const WbSwell *swell, const double *points,
                             const double *depths, const double *max_freqs,
                             int64_t n, double *out) {
  int64_t a;
  for (a = 0; a < n; a++)
    out[a] = wb_swell_height(swell, points + 3 * a, depths[a], max_freqs[a]);
}
