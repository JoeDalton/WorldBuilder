/* The planet clouds' billboards (world_builder.clouds.sphere, issue #20):
 * a tile's vertices at upload, and its back-to-front order at each re-sort.
 *
 * The order (_resort): the billboards sorted farthest first from the
 * camera, written straight as the tile's triangle indices.
 *
 * A stable LSD radix sort on the squared distance (float32 bits, which order
 * like the values for positive floats), over the previous order: equal
 * distances keep it, as numpy's stable argsort does in the fallback. Linear
 * in the billboards, where numpy's argsort and its index arithmetic cost
 * ~130 us per thousand. */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WB_API __declspec(dllexport)
#else
#define WB_API __attribute__((visibility("default")))
#endif

/* particles: (n, stride) float32 rows, xyz relative to the tile's centre
 * first; offset: the tile's centre minus the camera, (3,); order: (n,) the
 * previous order (quad indices, farthest first), updated in place; index:
 * (6 n,) the triangle indices, two triangles per quad (0 1 2, 0 2 3). */
WB_API void wb_billboard_order(const float *particles, int64_t stride, int64_t n,
                               const double *offset, uint32_t *order, uint32_t *index) {
  static const uint32_t QUAD[6] = {0, 1, 2, 0, 2, 3};
  uint32_t *block, *keys, *keys2, *order2, count[256];
  int64_t k;
  int pass, q;
  if (n <= 0) return;
  block = (uint32_t *)malloc((size_t)n * 3 * sizeof(uint32_t));
  if (!block) return;
  keys = block, keys2 = block + n, order2 = block + 2 * n;
  for (k = 0; k < n; k++) {
    const float *p = particles + (int64_t)order[k] * stride;
    double x = p[0] + offset[0], y = p[1] + offset[1], z = p[2] + offset[2];
    float d = (float)(x * x + y * y + z * z);
    uint32_t bits;
    memcpy(&bits, &d, sizeof(bits));
    keys[k] = ~bits; /* ascending keys: farthest first */
  }
  for (pass = 0; pass < 4; pass++) {
    int shift = 8 * pass;
    uint32_t total = 0, *tk, *to;
    memset(count, 0, sizeof(count));
    for (k = 0; k < n; k++) count[(keys[k] >> shift) & 0xffu]++;
    for (q = 0; q < 256; q++) {
      uint32_t c = count[q];
      count[q] = total;
      total += c;
    }
    for (k = 0; k < n; k++) {
      uint32_t slot = count[(keys[k] >> shift) & 0xffu]++;
      keys2[slot] = keys[k];
      order2[slot] = order[k];
    }
    tk = keys, keys = keys2, keys2 = tk;
    to = order2, order2 = order, order = to;
  }
  /* Four passes: the sorted data is back in the caller's buffers. */
  for (k = 0; k < n; k++) {
    uint32_t base = order[k] * 4u;
    for (q = 0; q < 6; q++) index[6 * k + q] = base + QUAD[q];
  }
  free(block);
}

/* clouds.sphere._tile_node's vertices: four per billboard (its corners),
 * each 11 float32 (sphere._vertex_format): xyz (the billboard's centre,
 * relative to the tile's), the corner (+-1, +-1), radius, sprite rect (u, v,
 * du, dv), layer. particles: (n, 9) rows, as CloudTile.particles; out: (4 n,
 * 11). box: (6,) the tile's bounds, min xyz then max xyz, each billboard's
 * centre widened by the largest radius; reach: the farthest a billboard
 * reaches from the tile's centre. */
WB_API void wb_billboard_vertices(const float *particles, int64_t n, float *out, float *box,
                                  double *reach) {
  static const float CORNERS[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
  float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
  float radius = 0.0f;
  double far2 = 0.0;
  int64_t k;
  int c, a;
  for (k = 0; k < n; k++) {
    const float *p = particles + 9 * k;
    double d2 = 0.0;
    for (c = 0; c < 4; c++) {
      float *v = out + 11 * (4 * k + c);
      v[0] = p[0], v[1] = p[1], v[2] = p[2];
      v[3] = CORNERS[c][0], v[4] = CORNERS[c][1];
      v[5] = p[3];
      v[6] = p[4], v[7] = p[5], v[8] = p[6], v[9] = p[7];
      v[10] = p[8];
    }
    for (a = 0; a < 3; a++) {
      if (p[a] < lo[a]) lo[a] = p[a];
      if (p[a] > hi[a]) hi[a] = p[a];
    }
    d2 = (double)p[0] * p[0] + (double)p[1] * p[1] + (double)p[2] * p[2];
    if (d2 > far2) far2 = d2;
    if (p[3] > radius) radius = p[3];
  }
  for (a = 0; a < 3; a++) {
    box[a] = lo[a] - radius;
    box[3 + a] = hi[a] + radius;
  }
  *reach = sqrt(far2) + radius;
}
