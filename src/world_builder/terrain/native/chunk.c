/* A chunk's geometry for world_builder.terrain.chunkgen (issue #20): the
 * steps around the planet's composition that only shape the grid, none of
 * its rules (those stay in Python).
 *
 * - wb_face_grid: cubesphere.face_dirs over a separable grid of one face
 *   (one tangent per column and per row instead of two per point);
 * - wb_chunk_surface: the inner grid's normals (chunkgen._grid_normals) and
 *   its slope at the biome scale (biomes.slope_from_normals);
 * - wb_chunk_pack: the vertex buffer (chunkgen._pack_vertices) and the box of
 *   what is drawn (chunkgen._sea_lifted).
 *
 * Each mirrors its numpy expression's order of operations, so it gives the
 * same values but where numpy's own tan and arccos differ from the C
 * library's by an ulp. tests/test_native_noise.py compares the two. */

#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(_WIN32)
#define WB_API __declspec(dllexport)
#else
#define WB_API __attribute__((visibility("default")))
#endif

#define QUADS 64
#define VERTS (QUADS + 1)
#define PI 3.141592653589793

/* (outward normal, u, v) per face: cubesphere.FACE_FRAMES. */
static const double FRAMES[6][3][3] = {
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},  {{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}},
    {{0, 1, 0}, {-1, 0, 0}, {0, 0, 1}}, {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
    {{0, 0, 1}, {0, 1, 0}, {-1, 0, 0}}, {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}},
};

/* cubesphere._warp: tan(pi a / 4), exactly +-1 at the face edges. */
static double warp(double a) {
  if (fabs(a) == 1.0) return a > 0.0 ? 1.0 : -1.0;
  return copysign(tan((PI / 4.0) * fabs(a)), a);
}

/* face_dirs(face, *meshgrid(a, b)): out (nb, na, 3), rows along b. */
WB_API void wb_face_grid(int32_t face, const double *a, int64_t na, const double *b,
                         int64_t nb, double *out) {
  const double *n = FRAMES[face][0], *u = FRAMES[face][1], *v = FRAMES[face][2];
  int64_t r, c;
  for (r = 0; r < nb; r++) {
    double y = warp(b[r]);
    for (c = 0; c < na; c++) {
      double x = warp(a[c]);
      double s = sqrt(1.0 + (x * x + y * y));
      double *p = out + 3 * (r * na + c);
      p[0] = (n[0] + x * u[0] + y * v[0]) / s;
      p[1] = (n[1] + x * u[1] + y * v[1]) / s;
      p[2] = (n[2] + x * u[2] + y * v[2]) / s;
    }
  }
}

/* The point at grid sample (r, c): dir * (radius + elevation). */
static void point_at(const double *dirs, const double *elevation, int64_t n, double radius,
                     int64_t r, int64_t c, double *p) {
  const double *d = dirs + 3 * (r * n + c);
  double h = radius + elevation[r * n + c];
  p[0] = d[0] * h, p[1] = d[1] * h, p[2] = d[2] * h;
}

/* _grid_normals: the outward unit normal at inner sample (r, c) by central
 * differences over +-s samples (u x v points out of the face). */
static void normal_at(const double *dirs, const double *elevation, int64_t n, double radius,
                      int64_t r, int64_t c, int32_t s, double *out) {
  double e[3], w[3], nn[3], su[3], sv[3], norm;
  int k;
  point_at(dirs, elevation, n, radius, r, c + s, e);
  point_at(dirs, elevation, n, radius, r, c - s, w);
  for (k = 0; k < 3; k++) su[k] = e[k] - w[k];
  point_at(dirs, elevation, n, radius, r + s, c, e);
  point_at(dirs, elevation, n, radius, r - s, c, w);
  for (k = 0; k < 3; k++) sv[k] = e[k] - w[k];
  nn[0] = su[1] * sv[2] - su[2] * sv[1];
  nn[1] = su[2] * sv[0] - su[0] * sv[2];
  nn[2] = su[0] * sv[1] - su[1] * sv[0];
  norm = sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
  for (k = 0; k < 3; k++) out[k] = nn[k] / norm;
}

/* dirs: (n, n, 3) and elevation: (n, n) of the sampled grid, the inner
 * VERTS x VERTS grid within `border` samples. normals: (VERTS^2, 3) by +-1
 * sample; slope: (VERTS^2,) degrees, from the normals by +-step samples. */
WB_API void wb_chunk_surface(const double *dirs, const double *elevation, int64_t n,
                             int32_t border, int32_t step, double radius, double *normals,
                             double *slope) {
  int64_t r, c;
  for (r = 0; r < VERTS; r++) {
    for (c = 0; c < VERTS; c++) {
      int64_t k = r * VERTS + c, gr = r + border, gc = c + border;
      const double *up = dirs + 3 * (gr * n + gc);
      double wide[3], cosine;
      normal_at(dirs, elevation, n, radius, gr, gc, 1, normals + 3 * k);
      normal_at(dirs, elevation, n, radius, gr, gc, step, wide);
      cosine = wide[0] * up[0] + wide[1] * up[1] + wide[2] * up[2];
      cosine = cosine < -1.0 ? -1.0 : (cosine > 1.0 ? 1.0 : cosine);
      slope[k] = acos(cosine) * (180.0 / PI);
    }
  }
}

/* chunkgen.skirt_ring: the edge vertices around the chunk, counter-clockwise
 * from the south-west corner. */
static int64_t ring_index(int64_t k) {
  int64_t side = k / QUADS, t = k % QUADS;
  if (side == 0) return t;                              /* south */
  if (side == 1) return t * VERTS + QUADS;              /* east */
  if (side == 2) return QUADS * VERTS + (QUADS - t);    /* north */
  return (QUADS - t) * VERTS;                           /* west */
}

/* chunkgen.VERTEX_DTYPE: 36 bytes. */
typedef struct {
  float vertex[3];
  float normal[3];
  float height;
  uint8_t biome_a[4];
  uint8_t biome_b[4];
} Vertex;

/* dirs, elevation, normals: the inner grid's ((VERTS^2, 3), (VERTS^2,),
 * (VERTS^2, 3)); weights: (VERTS^2, 6) bytes; centre: (3,). out: VERTS^2 +
 * 4 QUADS vertices, grid then skirt. box_min, box_max: (3,) the drawn
 * vertices' box, centre-relative; with sea, grid vertices under sea level
 * are lifted to it and the skirts hang from their lifted edge. */
WB_API void wb_chunk_pack(const double *dirs, const double *elevation, const double *normals,
                          const uint8_t *weights, const double *centre, double radius,
                          double skirt_depth, int32_t sea, uint8_t *out, float *box_min,
                          float *box_max) {
  Vertex *v = (Vertex *)out;
  const int64_t count = (int64_t)VERTS * VERTS;
  int64_t k;
  int a;
  float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
  if (sizeof(Vertex) != 36) return; /* never: no padding in this layout */
  for (k = 0; k < count; k++) {
    const double *d = dirs + 3 * k;
    double h = radius + elevation[k];
    float drawn[3];
    for (a = 0; a < 3; a++) {
      v[k].vertex[a] = (float)(d[a] * h - centre[a]);
      v[k].normal[a] = (float)normals[3 * k + a];
    }
    v[k].height = (float)elevation[k];
    memcpy(v[k].biome_a, weights + 6 * k, 4);
    v[k].biome_b[0] = weights[6 * k + 4], v[k].biome_b[1] = weights[6 * k + 5];
    v[k].biome_b[2] = 0, v[k].biome_b[3] = 0;
    for (a = 0; a < 3; a++)
      drawn[a] = sea && elevation[k] < 0.0 ? (float)(d[a] * radius - centre[a]) : v[k].vertex[a];
    for (a = 0; a < 3; a++) {
      if (drawn[a] < lo[a]) lo[a] = drawn[a];
      if (drawn[a] > hi[a]) hi[a] = drawn[a];
    }
  }
  for (k = 0; k < 4 * QUADS; k++) {
    int64_t e = ring_index(k);
    const double *d = dirs + 3 * e;
    double h = radius + elevation[e];
    Vertex *s = &v[count + k];
    float drawn[3];
    *s = v[e];
    for (a = 0; a < 3; a++) s->vertex[a] = (float)(d[a] * h - skirt_depth * d[a] - centre[a]);
    s->height = v[e].height - (float)skirt_depth;
    s->biome_b[3] = 255;
    for (a = 0; a < 3; a++) {
      drawn[a] = s->vertex[a];
      if (sea && elevation[e] < 0.0) drawn[a] = drawn[a] - (float)(elevation[e] * d[a]);
      if (drawn[a] < lo[a]) lo[a] = drawn[a];
      if (drawn[a] > hi[a]) hi[a] = drawn[a];
    }
  }
  memcpy(box_min, lo, sizeof(lo));
  memcpy(box_max, hi, sizeof(hi));
}
