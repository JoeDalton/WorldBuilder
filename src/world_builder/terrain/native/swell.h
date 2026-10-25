/* The sea's swell on the CPU: ocean/shaders/swell.glsl in double precision,
 * for the noise library (wb_swell_heights, which world_builder.ocean.swell
 * calls) and the collision solid, so collisions follow the drawn water. Plain
 * C99, usable from C++. Keep it in step with swell.glsl: the same lattice
 * hash (integer arithmetic, the same bits), the same octaves, the same
 * band limit. */
#ifndef WB_SWELL_H
#define WB_SWELL_H

#include <math.h>
#include <stdint.h>

/* The swell's parameters (world_builder.ocean.swell.Swell). The rotations are
 * row-major and apply as R . point. */
typedef struct {
  double amplitude; /* metres of water per unit of field, deep water */
  double scale;     /* the first layer's frequency, cycles per metre */
  double rot_a[9];
  double rot_b[9];
  int32_t by_depth; /* the swell dies out over the shallows */
  double shallow;   /* depth where it starts... */
  double deep;      /* ...and where it is whole, metres */
} WbSwell;

/* The field's mean, which faded octaves tend to (each layer's is 1/2). */
#define WB_SWELL_MEAN 0.25

static inline double wb_swell_lattice(int32_t x, int32_t y, int32_t z) {
  uint32_t h = ((uint32_t)x * 0x8da6b343u) ^ ((uint32_t)y * 0xd8163841u) ^
               ((uint32_t)z * 0xcb1ab31fu);
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  h *= 0x846ca68bu;
  h ^= h >> 16;
  return (double)(h >> 8) * (1.0 / 16777216.0);
}

static inline double wb_swell_mix(double a, double b, double t) {
  return a + (b - a) * t;
}

/* Value noise in [0, 1] at q (lattice units), smoothstep-interpolated. */
static inline double wb_swell_noise(double qx, double qy, double qz) {
  double ix = floor(qx), iy = floor(qy), iz = floor(qz);
  double fx = qx - ix, fy = qy - iy, fz = qz - iz;
  double ux = fx * fx * (3.0 - 2.0 * fx);
  double uy = fy * fy * (3.0 - 2.0 * fy);
  double uz = fz * fz * (3.0 - 2.0 * fz);
  int32_t x = (int32_t)ix, y = (int32_t)iy, z = (int32_t)iz;
  double x00 = wb_swell_mix(wb_swell_lattice(x, y, z), wb_swell_lattice(x + 1, y, z), ux);
  double x10 = wb_swell_mix(wb_swell_lattice(x, y + 1, z), wb_swell_lattice(x + 1, y + 1, z), ux);
  double x01 = wb_swell_mix(wb_swell_lattice(x, y, z + 1), wb_swell_lattice(x + 1, y, z + 1), ux);
  double x11 =
      wb_swell_mix(wb_swell_lattice(x, y + 1, z + 1), wb_swell_lattice(x + 1, y + 1, z + 1), ux);
  return wb_swell_mix(wb_swell_mix(x00, x10, uy), wb_swell_mix(x01, x11, uy), uz);
}

/* An octave's share at this frequency for a resolution: all of it below half
 * the cut-off frequency, none at it; max_freq <= 0 = no limit. */
static inline double wb_swell_band(double freq, double max_freq) {
  double t;
  if (max_freq <= 0.0) return 1.0;
  t = (2.0 * max_freq - 2.0 * freq) / max_freq;
  t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
  return t * t * (3.0 - 2.0 * t);
}

/* One layer: two octaves, faded toward their mean (1/2) by w0 and w1. */
static inline double wb_swell_layer(double qx, double qy, double qz, double w0, double w1) {
  double n0 = wb_swell_noise(qx, qy, qz);
  double n1 = wb_swell_noise(qx * 2.1 + 5.3, qy * 2.1 + 1.7, qz * 2.1 + 3.1);
  return 0.6 * wb_swell_mix(0.5, n0, w0) + 0.4 * wb_swell_mix(0.5, n1, w1);
}

/* The field at planet-frame point p (metres), in [0, 1], its octaves above
 * max_freq (cycles per metre) faded to their mean. */
static inline double wb_swell_field(const WbSwell *s, const double *p, double max_freq) {
  const double *a = s->rot_a, *b = s->rot_b;
  double k = s->scale, kb = 1.7 * s->scale;
  double ax = (a[0] * p[0] + a[1] * p[1] + a[2] * p[2]) * k;
  double ay = (a[3] * p[0] + a[4] * p[1] + a[5] * p[2]) * k;
  double az = (a[6] * p[0] + a[7] * p[1] + a[8] * p[2]) * k;
  double bx = (b[0] * p[0] + b[1] * p[1] + b[2] * p[2]) * kb;
  double by = (b[3] * p[0] + b[4] * p[1] + b[5] * p[2]) * kb;
  double bz = (b[6] * p[0] + b[7] * p[1] + b[8] * p[2]) * kb;
  double la = wb_swell_layer(ax, ay, az, wb_swell_band(k, max_freq),
                             wb_swell_band(2.1 * k, max_freq));
  double lb = wb_swell_layer(bx, by, bz, wb_swell_band(kb, max_freq),
                             wb_swell_band(3.57 * k, max_freq));
  return la * lb;
}

/* The water's height above sea level at sea-level point p over seabed this
 * deep (metres), for a resolution (max_freq: a chunk's 1 / (4 spacing)). */
static inline double wb_swell_height(const WbSwell *s, const double *p, double depth,
                                     double max_freq) {
  double factor = 1.0;
  if (s->amplitude == 0.0) return 0.0;
  if (s->by_depth) {
    double t = (depth - s->shallow) / (s->deep - s->shallow);
    t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    factor = t * t * (3.0 - 2.0 * t);
  }
  return s->amplitude * factor * (wb_swell_field(s, p, max_freq) - WB_SWELL_MEAN);
}

#endif
