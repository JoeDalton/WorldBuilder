// Asteroid field collision solid for Panda3D's CollisionTraverser (issue #14).
//
// One CollisionAsteroids solid stands for a field's rocks near the camera
// (registered from Python: world_builder.asteroids.collision). Each rock is
// one of the field's prototypes, turned and scaled: a closed cube-sphere
// mesh, the finest level of detail drawn (registered once per prototype).
// Spheres (ships, ordnance, sensors) and segments (lasers) are tested in
// each rock's own frame at the field's time, against those very triangles;
// the game's handlers get one CollisionEntry per collider per frame: the
// deepest sphere contact, the first segment hit.
//
// Frames: the solid lives in the collision node's frame: the field's axes,
// shifted by an origin (set_origin, float64) near the camera, so float32
// node coordinates stay small at any distance from the field's centre.
// Rocks and all maths are float64 in the field frame.
//
// Built into the same library as the terrain's solid (terrain_collision.cpp).
#define _USE_MATH_DEFINES  // M_PI with MSVC
#include "pandabase.h"
#include "collisionSolid.h"
#include "collisionEntry.h"
#include "collisionSphere.h"
#include "collisionSegment.h"
#include "collisionNode.h"
#include "omniBoundingVolume.h"
#include "referenceCount.h"
#include "register_type.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace {

struct V3 {
  double x, y, z;
  V3() : x(0), y(0), z(0) {}
  V3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
  V3 operator+(const V3 &o) const { return V3(x + o.x, y + o.y, z + o.z); }
  V3 operator-(const V3 &o) const { return V3(x - o.x, y - o.y, z - o.z); }
  V3 operator*(double s) const { return V3(x * s, y * s, z * s); }
  double dot(const V3 &o) const { return x * o.x + y * o.y + z * o.z; }
  V3 cross(const V3 &o) const {
    return V3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
  }
  double length() const { return std::sqrt(dot(*this)); }
  V3 normalized() const { double l = length(); return l > 0 ? *this * (1.0 / l) : *this; }
};

// A unit quaternion (w, x, y, z): motion.py's conventions.
struct Q {
  double w, x, y, z;
  Q() : w(1), x(0), y(0), z(0) {}
  Q(double w_, double x_, double y_, double z_) : w(w_), x(x_), y(y_), z(z_) {}
  Q operator*(const Q &b) const {
    return Q(w * b.w - x * b.x - y * b.y - z * b.z,
             w * b.x + x * b.w + y * b.z - z * b.y,
             w * b.y - x * b.z + y * b.w + z * b.x,
             w * b.z + x * b.y - y * b.x + z * b.w);
  }
  Q conj() const { return Q(w, -x, -y, -z); }
  V3 rotate(const V3 &v) const {
    V3 u(x, y, z);
    V3 t = u.cross(v) * 2.0;
    return v + t * w + u.cross(t);
  }
};

// motion.axis_angle: the turn by |w| * t about w.
Q turn(const V3 &w, double t) {
  double rate = w.length();
  if (rate < 1e-15) return Q();
  double half = 0.5 * rate * t;
  double s = std::sin(half) / rate;
  return Q(std::cos(half), w.x * s, w.y * s, w.z * s);
}

// (outward normal, u, v) per face: cubesphere.FACE_FRAMES.
const double FACE_FRAMES[6][3][3] = {
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
    {{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}},
    {{0, 1, 0}, {-1, 0, 0}, {0, 0, 1}},
    {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
    {{0, 0, 1}, {0, 1, 0}, {-1, 0, 0}},
    {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}},
};

inline double frame_dot(int face, int axis, const V3 &d) {
  const double *f = FACE_FRAMES[face][axis];
  return f[0] * d.x + f[1] * d.y + f[2] * d.z;
}

// A prototype's finest mesh: vertex positions per face, (n + 1)^2 each, row
// (v) major, bounding radius 1 (prototype.PrototypeMesh.vertices).
struct Prototype {
  int quads = 0;
  std::vector<V3> vertices;
  double spacing = 1.0;  // nominal grid spacing on the unit sphere

  const V3 &vertex(int face, int j, int i) const {
    return vertices[(size_t)face * (quads + 1) * (quads + 1) + (size_t)j * (quads + 1) + i];
  }

  // The radial ray along unit direction d against the triangle drawn there
  // (the meshes' diagonal): the surface's distance from the centre and the
  // triangle's outward normal.
  void surface(const V3 &d, double &radius, V3 &normal) const {
    double ax = std::fabs(d.x), ay = std::fabs(d.y), az = std::fabs(d.z);
    int axis = ax >= ay ? (ax >= az ? 0 : 2) : (ay >= az ? 1 : 2);
    double along = axis == 0 ? d.x : (axis == 1 ? d.y : d.z);
    int face = 2 * axis + (along < 0.0 ? 1 : 0);
    double n = frame_dot(face, 0, d);
    double a = std::atan(frame_dot(face, 1, d) / n) * (4.0 / M_PI);
    double b = std::atan(frame_dot(face, 2, d) / n) * (4.0 / M_PI);
    double x = (std::min(1.0, std::max(-1.0, a)) + 1.0) * 0.5 * quads;
    double y = (std::min(1.0, std::max(-1.0, b)) + 1.0) * 0.5 * quads;
    int i = std::min(quads - 1, std::max(0, (int)std::floor(x)));
    int j = std::min(quads - 1, std::max(0, (int)std::floor(y)));
    double fu = x - i, fv = y - j;
    const V3 &v00 = vertex(face, j, i);
    const V3 &v11 = vertex(face, j + 1, i + 1);
    const V3 &v1 = fu >= fv ? vertex(face, j, i + 1) : v11;
    const V3 &v2 = fu >= fv ? v11 : vertex(face, j + 1, i);
    V3 cr = (v1 - v00).cross(v2 - v00);
    double den = cr.dot(d);
    radius = std::fabs(den) > 1e-300 ? cr.dot(v00) / den : v00.length();
    normal = cr.normalized();
    if (normal.dot(d) < 0.0) normal = normal * -1.0;
  }
};

struct Rock {
  V3 centre;
  double radius;
  Q q0;
  V3 axis_rate;  // the axis times the precession rate
  double spin;
  int proto, cls;
  uint64_t key;  // its cell's key in its class
  bool live = false;

  // motion.orientation at time t.
  Q at(double t) const { return turn(axis_rate, t) * q0 * turn(V3(0, 0, spin), t); }
};

uint64_t cell_key(const V3 &p, double cell) {
  const int64_t off = (int64_t)1 << 20;
  uint64_t i = (uint64_t)((int64_t)std::floor(p.x / cell) + off);
  uint64_t j = (uint64_t)((int64_t)std::floor(p.y / cell) + off);
  uint64_t k = (uint64_t)((int64_t)std::floor(p.z / cell) + off);
  return (i << 42) | (j << 21) | k;
}

}  // namespace

class AsteroidData : public ReferenceCount {
public:
  AsteroidData(int classes, const double *cell, const double *hi)
      : cells(cell, cell + classes), his(hi, hi + classes), grid(classes), time(0.0) {}

  std::vector<double> cells, his;  // per size class: cell edge, largest radius
  std::vector<std::unordered_map<uint64_t, std::vector<int>>> grid;
  std::vector<Rock> rocks;  // by slot
  std::vector<Prototype> prototypes;
  double time;
  V3 origin;
  int count = 0;

  void add(int slot, const double *row) {
    if (slot >= (int)rocks.size()) rocks.resize(slot + 1);
    if (rocks[slot].live) remove(slot);
    Rock &r = rocks[slot];
    r.centre = V3(row[0], row[1], row[2]);
    r.radius = row[3];
    r.q0 = Q(row[4], row[5], row[6], row[7]);
    r.axis_rate = V3(row[8], row[9], row[10]) * row[11];
    r.spin = row[12];
    r.proto = (int)row[13];
    r.cls = std::min((int)row[14], (int)cells.size() - 1);
    r.key = cell_key(r.centre, cells[r.cls]);
    r.live = true;
    grid[r.cls][r.key].push_back(slot);
    ++count;
  }

  void remove(int slot) {
    if (slot < 0 || slot >= (int)rocks.size() || !rocks[slot].live) return;
    Rock &r = rocks[slot];
    auto it = grid[r.cls].find(r.key);
    if (it != grid[r.cls].end()) {
      std::vector<int> &v = it->second;
      v.erase(std::remove(v.begin(), v.end(), slot), v.end());
      if (v.empty()) grid[r.cls].erase(it);
    }
    r.live = false;
    --count;
  }

  // Every rock whose bounding sphere may meet the box [lo, hi].
  template <typename F>
  void near(const V3 &lo, const V3 &hi, F visit) const {
    const int64_t off = (int64_t)1 << 20;
    for (size_t k = 0; k < cells.size(); ++k) {
      if (grid[k].empty()) continue;
      double c = cells[k], reach = his[k];
      int64_t i0 = (int64_t)std::floor((lo.x - reach) / c), i1 = (int64_t)std::floor((hi.x + reach) / c);
      int64_t j0 = (int64_t)std::floor((lo.y - reach) / c), j1 = (int64_t)std::floor((hi.y + reach) / c);
      int64_t k0 = (int64_t)std::floor((lo.z - reach) / c), k1 = (int64_t)std::floor((hi.z + reach) / c);
      if ((i1 - i0 + 1) * (j1 - j0 + 1) * (k1 - k0 + 1) > (int64_t)grid[k].size()) {
        for (const auto &kv : grid[k])
          for (int slot : kv.second) visit(rocks[slot], slot);
        continue;
      }
      for (int64_t i = i0; i <= i1; ++i)
        for (int64_t j = j0; j <= j1; ++j)
          for (int64_t l = k0; l <= k1; ++l) {
            uint64_t key = ((uint64_t)(i + off) << 42) | ((uint64_t)(j + off) << 21) | (uint64_t)(l + off);
            auto it = grid[k].find(key);
            if (it == grid[k].end()) continue;
            for (int slot : it->second) visit(rocks[slot], slot);
          }
    }
  }

  // Sphere (c, r) against one rock, in its body frame (units of its
  // radius), as the terrain's sphere test: the nearest of a footprint of
  // samples, refined on its triangle's plane. Returns the penetration (r
  // minus the centre's distance to the surface; > 0 on contact).
  bool sphere_rock(const Rock &rock, const V3 &c, double r, int samples, V3 &point,
                   V3 &normal, double &depth) const {
    const Prototype &p = prototypes[rock.proto];
    if (p.quads == 0) return false;
    V3 rel = c - rock.centre;
    double reach = rock.radius + r;
    if (rel.dot(rel) >= reach * reach) return false;
    Q q = rock.at(time);
    double scale = 1.0 / rock.radius;
    V3 cb = q.conj().rotate(rel) * scale;
    double rb = r * scale;
    double rc = cb.length();
    V3 up = rc > 1e-12 ? cb * (1.0 / rc) : V3(0, 0, 1);
    V3 east = V3(0, 0, 1).cross(up);
    if (east.length() < 1e-6) east = V3(1, 0, 0).cross(up);
    east = east.normalized();
    V3 north = up.cross(east);
    double s;
    V3 n;
    double best = INFINITY;
    V3 best_point, best_normal;
    int half = samples / 2;
    double step = half > 0 ? std::min(rb, 1.0) / half : 0.0;
    for (int y = -half; y <= half; ++y) {
      for (int x = -half; x <= half; ++x) {
        V3 d = (up * std::max(rc, 1e-6) + east * (x * step) + north * (y * step)).normalized();
        p.surface(d, s, n);
        V3 sp = d * s;
        V3 diff = cb - sp;
        double d2 = diff.dot(diff);
        if (d2 < best) {
          best = d2;
          best_point = sp;
          best_normal = n;
        }
      }
    }
    V3 sp = best_point, sn = best_normal;
    for (int it = 0; it < 3; ++it) {
      V3 foot = cb - sn * (cb - sp).dot(sn);
      V3 d = foot.normalized();
      p.surface(d, s, n);
      V3 s2 = d * s;
      bool settled = (s2 - sp).dot(s2 - sp) < 1e-12;
      sp = s2;
      sn = n;
      if (settled) break;
    }
    double distance = (cb - sp).dot(sn);
    // Inside the rock along the radial: the centre is below the surface.
    p.surface(up, s, n);
    if (rc < s) distance = std::min(distance, rc - s);
    if (distance >= rb) return false;
    depth = (rb - distance) * rock.radius;
    V3 foot_b = cb - sn * distance;
    point = rock.centre + q.rotate(foot_b * rock.radius);
    normal = q.rotate(sn);
    return true;
  }

  // The deepest contact of a sphere with any rock.
  bool sphere(const V3 &c, double r, int samples, V3 &point, V3 &normal, double &depth,
              int &which) const {
    bool found = false;
    depth = -INFINITY;
    V3 lo = c - V3(r, r, r), hi = c + V3(r, r, r);
    near(lo, hi, [&](const Rock &rock, int slot) {
      V3 pt, nm;
      double d;
      if (sphere_rock(rock, c, r, samples, pt, nm, d) && d > depth) {
        depth = d;
        point = pt;
        normal = nm;
        which = slot;
        found = true;
      }
    });
    return found;
  }

  // The first crossing of segment a-b into one rock (t along a-b).
  bool segment_rock(const Rock &rock, const V3 &a, const V3 &b, double &t_hit, V3 &point,
                    V3 &normal) const {
    const Prototype &p = prototypes[rock.proto];
    if (p.quads == 0) return false;
    Q q = rock.at(time);
    double scale = 1.0 / rock.radius;
    V3 ab = q.conj().rotate(a - rock.centre) * scale;
    V3 bb = q.conj().rotate(b - rock.centre) * scale;
    V3 d = bb - ab;
    // Where the segment is within the bounding sphere (radius 1).
    double A = d.dot(d), B = 2.0 * ab.dot(d), C = ab.dot(ab) - 1.0;
    double t0, t1;
    if (A < 1e-300) {
      if (C > 0.0) return false;
      t0 = 0.0;
      t1 = 0.0;
    } else {
      double disc = B * B - 4.0 * A * C;
      if (disc < 0.0) return false;
      double sq = std::sqrt(disc);
      t0 = std::max(0.0, (-B - sq) / (2.0 * A));
      t1 = std::min(1.0, (-B + sq) / (2.0 * A));
      if (t0 > t1) return false;
    }
    double s;
    V3 n;
    auto inside = [&](double t) {
      V3 pt = ab + d * t;
      double r = pt.length();
      if (r < 1e-12) return -1.0;
      p.surface(pt * (1.0 / r), s, n);
      return r - s;
    };
    double length = std::sqrt(A);
    double step = 0.5 * p.spacing / std::max(length, 1e-12);
    int steps = std::max(1, (int)std::ceil((t1 - t0) / step));
    double prev = t0, hit = -1.0;
    if (inside(t0) <= 0.0) {
      hit = t0;
    } else {
      for (int k = 1; k <= steps; ++k) {
        double t = t0 + (t1 - t0) * k / steps;
        if (inside(t) <= 0.0) {
          double lo = prev, hi = t;
          for (int it = 0; it < 30; ++it) {
            double mid = 0.5 * (lo + hi);
            if (inside(mid) > 0.0) lo = mid; else hi = mid;
          }
          hit = hi;
          break;
        }
        prev = t;
      }
    }
    if (hit < 0.0) return false;
    V3 pt = ab + d * hit;
    V3 dir = pt.normalized();
    p.surface(dir, s, n);
    t_hit = hit;
    point = rock.centre + q.rotate(dir * (s * rock.radius));
    normal = q.rotate(n);
    return true;
  }

  // The first crossing of segment a-b into any rock.
  bool segment(const V3 &a, const V3 &b, double &t_hit, V3 &point, V3 &normal, int &which) const {
    bool found = false;
    t_hit = INFINITY;
    V3 lo(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
    V3 hi(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
    near(lo, hi, [&](const Rock &rock, int slot) {
      double t;
      V3 pt, nm;
      if (segment_rock(rock, a, b, t, pt, nm) && t < t_hit) {
        t_hit = t;
        point = pt;
        normal = nm;
        which = slot;
        found = true;
      }
    });
    return found;
  }
};

class CollisionAsteroids : public CollisionSolid {
public:
  explicit CollisionAsteroids(AsteroidData *data, int samples)
      : _data(data), _samples(std::max(1, samples | 1)) {}
  CollisionAsteroids(const CollisionAsteroids &copy)
      : CollisionSolid(copy), _data(copy._data), _samples(copy._samples) {}

  virtual CollisionSolid *make_copy() { return new CollisionAsteroids(*this); }
  virtual LPoint3 get_collision_origin() const { return LPoint3(0, 0, 0); }
  virtual void output(std::ostream &out) const {
    out << "asteroids(" << _data->count << " rocks)";
  }

protected:
  virtual PT(BoundingVolume) compute_internal_bounds() const {
    return new OmniBoundingVolume();
  }

  virtual PT(CollisionEntry) test_intersection_from_sphere(const CollisionEntry &entry) const {
    const CollisionSphere *sphere;
    DCAST_INTO_R(sphere, entry.get_from(), nullptr);
    const LMatrix4 &wrt = entry.get_wrt_mat();
    LPoint3 centre = sphere->get_center() * wrt;
    double r = length(LVector3(sphere->get_radius(), 0, 0) * wrt);
    const V3 &o = _data->origin;
    V3 c(centre[0] + o.x, centre[1] + o.y, centre[2] + o.z);
    V3 point, normal;
    double depth;
    int which;
    if (!_data->sphere(c, r, _samples, point, normal, depth, which)) return nullptr;
    PT(CollisionEntry) hit = new CollisionEntry(entry);
    hit->set_surface_normal(LVector3((PN_stdfloat)normal.x, (PN_stdfloat)normal.y, (PN_stdfloat)normal.z));
    hit->set_surface_point(LPoint3(point.x - o.x, point.y - o.y, point.z - o.z));
    V3 inside = c - normal * r;
    hit->set_interior_point(LPoint3(inside.x - o.x, inside.y - o.y, inside.z - o.z));
    return hit;
  }

  virtual PT(CollisionEntry) test_intersection_from_segment(const CollisionEntry &entry) const {
    const CollisionSegment *segment;
    DCAST_INTO_R(segment, entry.get_from(), nullptr);
    const LMatrix4 &wrt = entry.get_wrt_mat();
    LPoint3 la = segment->get_point_a() * wrt;
    LPoint3 lb = segment->get_point_b() * wrt;
    const V3 &o = _data->origin;
    V3 a(la[0] + o.x, la[1] + o.y, la[2] + o.z), b(lb[0] + o.x, lb[1] + o.y, lb[2] + o.z);
    double t;
    V3 point, normal;
    int which;
    if (!_data->segment(a, b, t, point, normal, which)) return nullptr;
    PT(CollisionEntry) hit = new CollisionEntry(entry);
    hit->set_surface_normal(LVector3((PN_stdfloat)normal.x, (PN_stdfloat)normal.y, (PN_stdfloat)normal.z));
    LPoint3 local(point.x - o.x, point.y - o.y, point.z - o.z);
    hit->set_surface_point(local);
    hit->set_interior_point(local);
    return hit;
  }

private:
  PT(AsteroidData) _data;
  int _samples;

public:
  static TypeHandle get_class_type() { return _type_handle; }
  static void init_type() {
    CollisionSolid::init_type();
    register_type(_type_handle, "CollisionAsteroids", CollisionSolid::get_class_type());
  }
  virtual TypeHandle get_type() const { return get_class_type(); }
  virtual TypeHandle force_init_type() { init_type(); return get_class_type(); }

private:
  static TypeHandle _type_handle;
};

TypeHandle CollisionAsteroids::_type_handle;

// ── C API (ctypes) ─────────────────────────────────────────────────────────────

#ifdef _WIN32
#define WB_API __declspec(dllexport)
#else
#define WB_API
#endif

extern "C" {

WB_API int wb_asteroids_init() {
  CollisionAsteroids::init_type();
  return (int)CollisionAsteroids::get_class_type().get_index();
}

// classes: size classes; cell: (classes,) cell edges; hi: largest radii.
WB_API void *wb_asteroids_new(int classes, const double *cell, const double *hi) {
  AsteroidData *data = new AsteroidData(classes, cell, hi);
  data->ref();
  return data;
}

WB_API void wb_asteroids_free(void *handle) {
  unref_delete((AsteroidData *)handle);
}

// A prototype's finest mesh: 6 (quads + 1)^2 float32 positions, x y z.
WB_API void wb_asteroids_set_prototype(void *handle, int proto, int quads, const float *vertices) {
  AsteroidData *data = (AsteroidData *)handle;
  if (proto >= (int)data->prototypes.size()) data->prototypes.resize(proto + 1);
  Prototype &p = data->prototypes[proto];
  p.quads = quads;
  p.spacing = 0.5 * M_PI / quads;
  size_t n = (size_t)6 * (quads + 1) * (quads + 1);
  p.vertices.resize(n);
  for (size_t k = 0; k < n; ++k)
    p.vertices[k] = V3(vertices[3 * k], vertices[3 * k + 1], vertices[3 * k + 2]);
}

// Rocks by slot: per row centre (3), radius, q0 (4), axis (3), precession,
// spin, prototype, class (15 doubles, field frame).
WB_API void wb_asteroids_add(void *handle, long long n, const long long *slots, const double *rows) {
  AsteroidData *data = (AsteroidData *)handle;
  for (long long k = 0; k < n; ++k) data->add((int)slots[k], rows + 15 * k);
}

WB_API void wb_asteroids_remove(void *handle, long long n, const long long *slots) {
  AsteroidData *data = (AsteroidData *)handle;
  for (long long k = 0; k < n; ++k) data->remove((int)slots[k]);
}

WB_API int wb_asteroids_count(void *handle) { return ((AsteroidData *)handle)->count; }

// The field's time (the rocks' orientations), seconds.
WB_API void wb_asteroids_set_time(void *handle, double t) { ((AsteroidData *)handle)->time = t; }

// The field-frame point at the collision node's origin.
WB_API void wb_asteroids_set_origin(void *handle, double x, double y, double z) {
  ((AsteroidData *)handle)->origin = V3(x, y, z);
}

WB_API int wb_asteroids_attach(void *handle, void *node, int samples) {
  CollisionNode *collision_node = (CollisionNode *)node;
  collision_node->add_solid(new CollisionAsteroids((AsteroidData *)handle, samples));
  return 1;
}

// Field-frame queries, for tests: out = point (3), normal (3), depth, slot.
WB_API int wb_asteroids_sphere(void *handle, const double *c, double r, int samples, double *out) {
  V3 point, normal;
  double depth;
  int which = -1;
  if (!((AsteroidData *)handle)->sphere(V3(c[0], c[1], c[2]), r, std::max(1, samples | 1), point,
                                         normal, depth, which))
    return 0;
  double values[8] = {point.x, point.y, point.z, normal.x, normal.y, normal.z, depth, (double)which};
  std::copy(values, values + 8, out);
  return 1;
}

// out = t, point (3), normal (3), slot.
WB_API int wb_asteroids_segment(void *handle, const double *a, const double *b, double *out) {
  double t;
  V3 point, normal;
  int which = -1;
  if (!((AsteroidData *)handle)->segment(V3(a[0], a[1], a[2]), V3(b[0], b[1], b[2]), t, point,
                                          normal, which))
    return 0;
  double values[8] = {t, point.x, point.y, point.z, normal.x, normal.y, normal.z, (double)which};
  std::copy(values, values + 8, out);
  return 1;
}

}  // extern "C"
