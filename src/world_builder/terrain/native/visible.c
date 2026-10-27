/* Asteroid fields' visible pass (world_builder.asteroids.population
 * Population.visible, issue #14): per frame, over the rocks held, those big
 * enough on screen and within their class's limit, inside some view, at the
 * coarsest level of detail whose error stays under the limit (with
 * hysteresis), grouped by (prototype, level). The numpy reference's
 * arithmetic in its order, so both choose the same rocks.
 *
 * Plain C99; built into the noise library without floating-point
 * contraction (scripts/build_native.sh). */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WB_API __declspec(dllexport)
#else
#define WB_API __attribute__((visibility("default")))
#endif

/* population.SLOT_DTYPE: what the pass reads of a rock held. */
typedef struct {
  double centre[3];
  double radius;
  int32_t cls;
  int32_t proto;
} WbSlot;

/* slots: (n,) what is held of each slot; live: (n,) held; usable: (n,) held,
 * on the GPU and not drawn as terrain; lod: (n,) each slot's level, in and
 * out (-1: none yet); planes: (views, 4, 3) inward normals through the
 * camera; errors: (prototypes, levels) in units of the radius; limits:
 * (classes,) no rock of a class farther. out: (n,) the slots drawn, grouped
 * by bucket (prototype * levels + level); counts: (prototypes * levels,).
 * near: (n,) the live slots of radius >= near_radius closer than
 * near_factor radii, their count in *near_count. Returns the slots drawn
 * (-1: out of memory), the rocks in range (any view) in *in_range. */
WB_API int64_t wb_asteroid_visible(
    const WbSlot *slots, const uint8_t *live, const uint8_t *usable, int64_t n,
    int8_t *lod, const double *camera, const double *planes, int32_t views,
    double ppr, double min_px, double lod_error_px, double hysteresis,
    const double *errors, int32_t prototypes, int32_t levels,
    const double *limits, double near_radius, double near_factor,
    int64_t *near, int64_t *near_count, int64_t *out, int64_t *counts,
    int64_t *in_range) {
  int64_t drawn = 0, range = 0, close = 0, i, b,
          buckets = (int64_t)prototypes * levels;
  int32_t *bucket = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n > 0 ? n : 1));
  int64_t *slot = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n > 0 ? n : 1));
  if (!bucket || !slot) {
    free(bucket);
    free(slot);
    return -1;
  }
  for (b = 0; b < buckets; b++) counts[b] = 0;
  for (i = 0; i < n; i++) {
    const WbSlot *r;
    double x, y, z, dist, scale;
    int inside = 0, v, k, l, chosen, previous;
    if (!live[i]) continue;
    r = slots + i;
    x = r->centre[0] - camera[0];
    y = r->centre[1] - camera[1];
    z = r->centre[2] - camera[2];
    dist = sqrt(x * x + y * y + z * z);
    if (r->radius >= near_radius && dist < near_factor * r->radius)
      near[close++] = i;
    if (!usable[i]) continue;
    scale = r->radius * ppr / (dist > 1e-6 ? dist : 1e-6);
    if (!(2.0 * scale >= min_px) || !(dist <= limits[r->cls])) continue;
    range++;
    for (v = 0; v < views && !inside; v++) {
      const double *p = planes + 12 * v;
      inside = 1;
      for (k = 0; k < 4; k++) {
        const double *m = p + 3 * k;
        if (x * m[0] + y * m[1] + z * m[2] < -r->radius) {
          inside = 0;
          break;
        }
      }
    }
    if (!inside) continue;
    previous = lod[i];
    chosen = levels - 1;
    for (l = 0; l < levels; l++) {
      double pixels = errors[(int64_t)r->proto * levels + l] * scale;
      int fits = pixels <= lod_error_px;
      if (l < previous) fits = fits && pixels * hysteresis <= lod_error_px;
      if (l == levels - 1) fits = 1;
      if (fits) {
        chosen = l;
        break;
      }
    }
    lod[i] = (int8_t)chosen;
    bucket[drawn] = r->proto * levels + chosen;
    slot[drawn] = i;
    counts[bucket[drawn]]++;
    drawn++;
  }
  /* A stable counting sort by bucket: slots stay in order within one. */
  {
    int64_t *next = (int64_t *)malloc(sizeof(int64_t) * (size_t)(buckets > 0 ? buckets : 1));
    int64_t at = 0;
    if (!next) {
      free(bucket);
      free(slot);
      return -1;
    }
    for (b = 0; b < buckets; b++) {
      next[b] = at;
      at += counts[b];
    }
    for (i = 0; i < drawn; i++) out[next[bucket[i]]++] = slot[i];
    free(next);
  }
  free(bucket);
  free(slot);
  *in_range = range;
  *near_count = close;
  return drawn;
}
