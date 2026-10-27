/* Asteroid fields' rocks (world_builder.asteroids.scatter, issue #14).
 *
 * The numpy reference's steps for the cells asked for: per cell, a Poisson
 * count of candidates drawn from the cell's hash; per candidate, its
 * position, radius and existence (the density there: the field's profile
 * times the clumping noise; no keep-out entered); then the hard core (it is
 * dropped where it overlaps an existing candidate that ranks above it, of
 * its own class in the 27 cells around, or of any larger class around its
 * position); then the survivors' attributes. Every choice uses the same
 * arithmetic, in the same order, on the same values as numpy, so both give
 * the same rocks; the orientations and axes go through sin, cos and log and
 * may differ in the last bits.
 *
 * Plain C99; built into the noise library without floating-point
 * contraction (scripts/build_native.sh). */
#include <math.h>
#include <stdint.h>

#include "noise.h"

#if defined(_WIN32)
#define WB_API __declspec(dllexport)
#else
#define WB_API __attribute__((visibility("default")))
#endif

#define TWO_PI 6.283185307179586
#define PI 3.141592653589793

/* scatter.py's draw slots, per candidate after the cell's count (slot 0). */
#define SLOTS 20
enum {
  POS = 0, THIN = 3, SIZE = 4, ID = 5, PROTO = 6, ICE = 7, RATE = 8,
  AXIS_PHI = 9, AXIS_W = 10, TUMBLE = 11, CONE = 12, RATIO = 13,
  AZIMUTH = 14, PHASE = 15, Q0 = 16
};

/* scatter._WbScatter */
typedef struct {
  int32_t shape; /* 0 box, 1 sphere, 2 annulus */
  int32_t classes;
  double half[3]; /* box half sizes; sphere radius in [0]; annulus inner,
                     outer, half thickness */
  double edge;
  int32_t clump_octaves; /* 0 = no clumping */
  int32_t poisson_max;
  const double *clump_frames;
  const uint64_t *clump_seeds;
  const double *clump_weights;
  double clump_norm, clump_low, clump_width;
  const double *class_lo, *class_hi, *class_cell;
  const uint64_t *class_seed;
  const double *poisson; /* (classes, poisson_max) cumulative */
  int32_t table_n;
  int32_t prototypes;
  /* size_table: (classes, table_n), each class's own sizes */
  const double *size_table, *ice_table, *rate_table, *cone_table, *ratio_table;
  double rate_ref, size_exponent;
  double axis_frame[9]; /* e1, e2, axis (rows) */
  double kappa;         /* 0 = isotropic */
  double tumble_fraction;
  int32_t keep_outs;
  const double *keep_out; /* (keep_outs, 7): a, b, clearance */
  double radial_edge;      /* annulus */
  int32_t profile_n;       /* annulus: 0 = even */
  const double *profile;
} WbScatter;

/* A candidate before the hard core. */
typedef struct {
  uint64_t h, base, id;
  double centre[3];
  double radius;
  int alive;
} Candidate;

static uint64_t draw(uint64_t h, uint64_t index) {
  return wb_mix64(h + index * 0x9E3779B97F4A7C15ULL);
}

static double unit(uint64_t bits) {
  return (double)(bits >> 11) * 1.1102230246251565e-16; /* 2^-53 */
}

/* distributions.interpolate */
static double table_lerp(const double *table, int32_t n, double u) {
  int32_t last = n - 1;
  double x = u * last;
  int64_t i = (int64_t)floor(x);
  double f;
  if (i > last - 1) i = last - 1;
  f = x - (double)i;
  return table[i] + (table[i + 1] - table[i]) * f;
}

static double smoothstep01(double t) {
  if (t < 0.0) t = 0.0;
  if (t > 1.0) t = 1.0;
  return t * t * (3.0 - 2.0 * t);
}

/* _Tables.density */
static double density(const WbScatter *s, const double *p) {
  double profile, clump = 1.0;
  if (s->shape == 0) {
    if (s->edge > 0.0) {
      double sx = smoothstep01((s->half[0] - fabs(p[0])) / s->edge);
      double sy = smoothstep01((s->half[1] - fabs(p[1])) / s->edge);
      double sz = smoothstep01((s->half[2] - fabs(p[2])) / s->edge);
      profile = sx * sy * sz;
    } else {
      profile = (fabs(p[0]) <= s->half[0] && fabs(p[1]) <= s->half[1] &&
                 fabs(p[2]) <= s->half[2])
                    ? 1.0
                    : 0.0;
    }
  } else if (s->shape == 1) {
    double d = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    if (s->edge > 0.0)
      profile = smoothstep01((s->half[0] - d) / s->edge);
    else
      profile = d <= s->half[0] ? 1.0 : 0.0;
  } else {
    /* spec.Annulus.vertical * spec.Annulus.radial */
    double rho = sqrt(p[0] * p[0] + p[1] * p[1]);
    double vertical, radial;
    if (s->edge > 0.0)
      vertical = smoothstep01((s->half[2] - fabs(p[2])) / s->edge);
    else
      vertical = fabs(p[2]) <= s->half[2] ? 1.0 : 0.0;
    if (s->radial_edge > 0.0)
      radial = smoothstep01((rho - s->half[0]) / s->radial_edge) *
               smoothstep01((s->half[1] - rho) / s->radial_edge);
    else
      radial = (rho >= s->half[0] && rho <= s->half[1]) ? 1.0 : 0.0;
    if (s->profile_n > 0) {
      double u = (rho - s->half[0]) / (s->half[1] - s->half[0]);
      if (u < 0.0) u = 0.0;
      if (u > 1.0) u = 1.0;
      radial = radial * table_lerp(s->profile, s->profile_n, u);
    }
    profile = vertical * radial;
  }
  if (s->clump_octaves > 0 && profile > 0.0) {
    double v = wb_fbm_point(p, s->clump_octaves, s->clump_frames,
                            s->clump_seeds, s->clump_weights) /
               s->clump_norm;
    clump = smoothstep01((v - s->clump_low) / s->clump_width);
  } else if (s->clump_octaves > 0) {
    clump = 0.0; /* profile * clump is 0 whatever the noise */
  }
  return profile * clump;
}

/* _Tables.kept_out */
static int kept_out(const WbScatter *s, const double *c, double radius) {
  int32_t k;
  for (k = 0; k < s->keep_outs; k++) {
    const double *row = s->keep_out + 7 * k;
    double ab[3], d[3], length2, dist2, reach;
    ab[0] = row[3] - row[0];
    ab[1] = row[4] - row[1];
    ab[2] = row[5] - row[2];
    d[0] = c[0] - row[0];
    d[1] = c[1] - row[1];
    d[2] = c[2] - row[2];
    length2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    if (length2 > 0.0) {
      double t = (d[0] * ab[0] + d[1] * ab[1] + d[2] * ab[2]) / length2;
      if (t < 0.0) t = 0.0;
      if (t > 1.0) t = 1.0;
      d[0] = d[0] - t * ab[0];
      d[1] = d[1] - t * ab[1];
      d[2] = d[2] - t * ab[2];
    }
    dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    reach = row[6] + radius;
    if (dist2 < reach * reach) return 1;
  }
  return 0;
}

/* The cell's candidate count (scatter._candidates). */
static int32_t cell_count(const WbScatter *s, int32_t cls, uint64_t h) {
  const double *cum = s->poisson + (int64_t)cls * s->poisson_max;
  double u = unit(draw(h, 0));
  int32_t n = 0;
  while (n < s->poisson_max && u >= cum[n]) n++;
  return n;
}

/* scatter._cell_hash: splitmix64 chained over the cell's indices (not the
 * noise's lattice hash, which collides on small coordinates). */
static uint64_t cell_hash(const WbScatter *s, int32_t cls, const int64_t *cell) {
  uint64_t h = wb_mix64(s->class_seed[cls] + (uint64_t)cell[0]);
  h = wb_mix64(h + (uint64_t)cell[1]);
  return wb_mix64(h + (uint64_t)cell[2]);
}

/* Candidate m of a cell: its position and radius always, its existence
 * only if `alive` is asked for (the density costs the noise). */
static void candidate(const WbScatter *s, int32_t cls, const int64_t *cell,
                      uint64_t h, int32_t m, int with_alive, Candidate *c) {
  double cellsize = s->class_cell[cls];
  int a;
  c->h = h;
  c->base = 1 + (uint64_t)m * SLOTS;
  for (a = 0; a < 3; a++)
    c->centre[a] =
        ((double)cell[a] + unit(draw(h, c->base + POS + a))) * cellsize;
  c->radius = table_lerp(s->size_table + (int64_t)cls * s->table_n, s->table_n,
                         unit(draw(h, c->base + SIZE)));
  c->id = draw(h, c->base + ID);
  c->alive = -1;
  if (with_alive) {
    c->alive = unit(draw(h, c->base + THIN)) < density(s, c->centre);
    if (c->alive && s->keep_outs > 0) c->alive = !kept_out(s, c->centre, c->radius);
  }
}

/* Whether candidate b, if it exists, outranks and overlaps a. */
static int outranks_and_touches(const Candidate *a, const Candidate *b) {
  double d[3], dist2, reach;
  if (b->id == a->id) return 0;
  if (!(b->radius > a->radius || (b->radius == a->radius && b->id < a->id)))
    return 0;
  d[0] = b->centre[0] - a->centre[0];
  d[1] = b->centre[1] - a->centre[1];
  d[2] = b->centre[2] - a->centre[2];
  dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  reach = a->radius + b->radius;
  return dist2 < reach * reach;
}

/* Whether any existing candidate of class `cls` in the 27 cells around
 * `around` outranks and overlaps a. */
static int blocked_in(const WbScatter *s, int32_t cls, const int64_t *around,
                      const Candidate *a) {
  int dx, dy, dz;
  for (dx = -1; dx <= 1; dx++)
    for (dy = -1; dy <= 1; dy++)
      for (dz = -1; dz <= 1; dz++) {
        int64_t cell[3];
        uint64_t h;
        int32_t n, m;
        cell[0] = around[0] + dx;
        cell[1] = around[1] + dy;
        cell[2] = around[2] + dz;
        h = cell_hash(s, cls, cell);
        n = cell_count(s, cls, h);
        for (m = 0; m < n; m++) {
          Candidate b;
          candidate(s, cls, cell, h, m, 0, &b);
          if (!outranks_and_touches(a, &b)) continue;
          candidate(s, cls, cell, h, m, 1, &b);
          if (b.alive) return 1;
        }
      }
  return 0;
}

static void quat_multiply(const double *a, const double *b, double *out) {
  out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

/* scatter._cone_orientation */
static void cone_orientation(const double *axis, double cone, double azimuth,
                             double phase, double *q) {
  double helper[3] = {1.0, 0.0, 0.0}, f1[3], f2[3], zb[3], align[4], turn[4];
  double n, alpha = TWO_PI * azimuth, sc = sin(cone), cc = cos(cone), half;
  int a;
  if (!(fabs(axis[0]) < 0.9)) {
    helper[0] = 0.0;
    helper[1] = 1.0;
  }
  f1[0] = axis[1] * helper[2] - axis[2] * helper[1];
  f1[1] = axis[2] * helper[0] - axis[0] * helper[2];
  f1[2] = axis[0] * helper[1] - axis[1] * helper[0];
  n = sqrt(f1[0] * f1[0] + f1[1] * f1[1] + f1[2] * f1[2]);
  for (a = 0; a < 3; a++) f1[a] /= n;
  f2[0] = axis[1] * f1[2] - axis[2] * f1[1];
  f2[1] = axis[2] * f1[0] - axis[0] * f1[2];
  f2[2] = axis[0] * f1[1] - axis[1] * f1[0];
  for (a = 0; a < 3; a++)
    zb[a] = cc * axis[a] + sc * cos(alpha) * f1[a] + sc * sin(alpha) * f2[a];
  align[0] = 1.0 + zb[2];
  align[1] = -zb[1];
  align[2] = zb[0];
  align[3] = 0.0;
  if (align[0] < 1e-12) {
    align[0] = 0.0;
    align[1] = 1.0;
    align[2] = 0.0;
  } else {
    n = sqrt(align[0] * align[0] + align[1] * align[1] + align[2] * align[2] +
             align[3] * align[3]);
    for (a = 0; a < 4; a++) align[a] /= n;
  }
  half = PI * phase;
  turn[0] = cos(half);
  turn[1] = 0.0;
  turn[2] = 0.0;
  turn[3] = sin(half);
  quat_multiply(align, turn, q);
}

/* scatter._attributes, for one surviving candidate. */
static void attributes(const WbScatter *s, int32_t cls, const Candidate *c,
                       WbRock *r) {
  uint64_t h = c->h, b = c->base;
  double rate, phi, v, w, sw, axis[3], cone, ratio;
  const double *e1 = s->axis_frame, *e2 = s->axis_frame + 3,
               *ax = s->axis_frame + 6;
  int a, tumbling;
  r->id = c->id;
  r->cls = cls;
  r->centre[0] = c->centre[0];
  r->centre[1] = c->centre[1];
  r->centre[2] = c->centre[2];
  r->radius = c->radius;
  r->proto = (int32_t)(draw(h, b + PROTO) % (uint64_t)s->prototypes);
  r->ice = table_lerp(s->ice_table, s->table_n, unit(draw(h, b + ICE)));
  rate = table_lerp(s->rate_table, s->table_n, unit(draw(h, b + RATE)));
  rate = rate * pow(c->radius / s->rate_ref, -s->size_exponent);
  phi = TWO_PI * unit(draw(h, b + AXIS_PHI));
  v = unit(draw(h, b + AXIS_W));
  if (s->kappa > 0.0)
    w = 1.0 + log(v + (1.0 - v) * exp(-2.0 * s->kappa)) / s->kappa;
  else
    w = 2.0 * v - 1.0;
  sw = 1.0 - w * w;
  sw = sqrt(sw > 0.0 ? sw : 0.0);
  for (a = 0; a < 3; a++)
    axis[a] = sw * cos(phi) * e1[a] + sw * sin(phi) * e2[a] + w * ax[a];
  r->axis[0] = axis[0];
  r->axis[1] = axis[1];
  r->axis[2] = axis[2];
  r->precession = rate;
  tumbling = unit(draw(h, b + TUMBLE)) < s->tumble_fraction;
  cone = table_lerp(s->cone_table, s->table_n, unit(draw(h, b + CONE)));
  ratio = table_lerp(s->ratio_table, s->table_n, unit(draw(h, b + RATIO)));
  if (tumbling) {
    r->spin = rate * cos(cone) * ratio;
    cone_orientation(axis, cone, unit(draw(h, b + AZIMUTH)),
                     unit(draw(h, b + PHASE)), r->q0);
  } else {
    double u1 = unit(draw(h, b + Q0)), u2 = unit(draw(h, b + Q0 + 1)),
           u3 = unit(draw(h, b + Q0 + 2));
    double ra = sqrt(1.0 - u1), rb = sqrt(u1);
    r->spin = 0.0;
    r->q0[0] = rb * cos(TWO_PI * u3);
    r->q0[1] = ra * sin(TWO_PI * u2);
    r->q0[2] = ra * cos(TWO_PI * u2);
    r->q0[3] = rb * sin(TWO_PI * u3);
  }
}

/* The rocks of class `cls` in `cells` ((n, 3) cell indices), in the cells'
 * order then the candidates'. Writes at most `capacity` into `out` and
 * returns how many there are (more than capacity: call again with room). */
WB_API int64_t wb_scatter_cells(const WbScatter *s, int32_t cls,
                                const int64_t *cells, int64_t n, WbRock *out,
                                int64_t capacity) {
  int64_t a, found = 0;
  for (a = 0; a < n; a++) {
    const int64_t *cell = cells + 3 * a;
    uint64_t h = cell_hash(s, cls, cell);
    int32_t count = cell_count(s, cls, h), m;
    for (m = 0; m < count; m++) {
      Candidate c;
      int32_t j, blocked;
      candidate(s, cls, cell, h, m, 1, &c);
      if (!c.alive) continue;
      blocked = blocked_in(s, cls, cell, &c);
      for (j = cls + 1; j < s->classes && !blocked; j++) {
        int64_t own[3];
        int k;
        for (k = 0; k < 3; k++)
          own[k] = (int64_t)floor(c.centre[k] / s->class_cell[j]);
        blocked = blocked_in(s, j, own, &c);
      }
      if (blocked) continue;
      if (found < capacity) attributes(s, cls, &c, out + found);
      found++;
    }
  }
  return found;
}
