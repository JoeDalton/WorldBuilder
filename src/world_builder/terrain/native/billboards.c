/* The planet clouds' back-to-front order (world_builder.clouds.sphere's
 * _resort, issue #20): a tile's billboards sorted farthest first from the
 * camera, written straight as the tile's triangle indices.
 *
 * A stable LSD radix sort on the squared distance (float32 bits, which order
 * like the values for positive floats), over the previous order: equal
 * distances keep it, as numpy's stable argsort does in the fallback. Linear
 * in the billboards, where numpy's argsort and its index arithmetic cost
 * ~130 us per thousand. */

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
