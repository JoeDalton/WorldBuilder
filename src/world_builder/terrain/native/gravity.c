/* Native gravity field for world_builder.gravity.
 *
 * g(x) = sum_i gm_i * (p_i - x) / max(|p_i - x|, radius_i)^3
 *
 * One pass over the points, the bodies in the inner loop: no temporaries,
 * one read of the points and one write of the result. The same sum, in the
 * same order, as the numpy version. bodies: per body x, y, z, G*mass, radius
 * squared (5 doubles). */

#include <math.h>
#include <stdint.h>

#if defined(_WIN32)
#define WB_API __declspec(dllexport)
#else
#define WB_API __attribute__((visibility("default")))
#endif

WB_API void wb_gravity(const double *points, int64_t n, const double *bodies,
                       int32_t count, double *out) {
  for (int64_t i = 0; i < n; i++) {
    const double x = points[3 * i], y = points[3 * i + 1], z = points[3 * i + 2];
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int32_t b = 0; b < count; b++) {
      const double *body = bodies + 5 * b;
      const double dx = body[0] - x, dy = body[1] - y, dz = body[2] - z;
      double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < body[4]) r2 = body[4];
      const double k = body[3] / (r2 * sqrt(r2));
      ax += dx * k;
      ay += dy * k;
      az += dz * k;
    }
    out[3 * i] = ax;
    out[3 * i + 1] = ay;
    out[3 * i + 2] = az;
  }
}
