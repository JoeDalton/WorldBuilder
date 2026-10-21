// Terrain collision solid for Panda3D's CollisionTraverser (issue #16).
//
// One CollisionTerrain solid stands for a whole planet's drawn terrain. It
// holds the grid vertices of the chunks the renderer shows (registered from
// Python) and answers sphere and segment tests against the very triangles on
// screen (same vertices, same diagonal as query.py). The game's handlers get
// ordinary CollisionEntry objects: one per collider per frame.
//
// Frames: the solid lives in the terrain root's frame, which is the planet
// frame shifted by the root's snapped origin (Terrain._rebase); set_origin
// keeps it in sync. All maths run in float64 in the planet frame.
#include "pandabase.h"
#include "collisionSolid.h"
#include "collisionEntry.h"
#include "collisionSphere.h"
#include "collisionSegment.h"
#include "collisionNode.h"
#include "omniBoundingVolume.h"
#include "referenceCount.h"
#include "register_type.h"
#include "pandaVersion.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

const int QUADS = 64;
const int VERTS = QUADS + 1;
const int GRID = VERTS * VERTS;
const int BIOMES = 6;
// Height bounds are kept per tile of TILE x TILE quads.
const int TILE = 8;
const int TILES = QUADS / TILE;

// (outward normal, u, v) per face: +X, -X, +Y, -Y, +Z, -Z (cubesphere.FACE_FRAMES).
const double FACE_FRAMES[6][3][3] = {
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
    {{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}},
    {{0, 1, 0}, {-1, 0, 0}, {0, 0, 1}},
    {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
    {{0, 0, 1}, {0, 1, 0}, {-1, 0, 0}},
    {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}},
};

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

inline double frame_dot(int face, int axis, const V3 &d) {
  const double *f = FACE_FRAMES[face][axis];
  return f[0] * d.x + f[1] * d.y + f[2] * d.z;
}

struct Chunk {
  int level, face;
  int64_t i, j;
  V3 centre;
  float h_min, h_max;
  // Highest ground of each tile and its 8 neighbours (sea level included
  // with an ocean): any surface within one tile of a point is below it.
  float tile_top[TILES][TILES];
  std::vector<float> pos;          // GRID * 3, relative to centre
  std::vector<unsigned char> weights;  // GRID * BIOMES
};

inline uint64_t pack(int level, int face, int64_t i, int64_t j) {
  return ((uint64_t)level << 56) | ((uint64_t)face << 48) | ((uint64_t)i << 24) | (uint64_t)j;
}

// A point on the surface: height above sea level along the radial, the
// triangle's outward normal, and the chunk that answered.
struct Surface {
  double height;
  V3 normal;
  const Chunk *chunk;
};

}  // namespace

class TerrainData : public ReferenceCount {
public:
  TerrainData(double radius, int max_level, int64_t n0, bool ocean)
      : radius(radius), max_level(max_level), n0(n0), ocean(ocean),
        spacing0(M_PI * radius / 2.0 / (double)n0), finest(max_level) {}
  ~TerrainData() { for (auto &kv : chunks) delete kv.second; }

  double radius;
  int max_level;
  int64_t n0;
  bool ocean;
  double spacing0;
  int finest;
  V3 origin;
  std::unordered_map<uint64_t, Chunk *> chunks;

  void add(Chunk *chunk) {
    uint64_t key = pack(chunk->level, chunk->face, chunk->i, chunk->j);
    auto it = chunks.find(key);
    if (it != chunks.end()) delete it->second;
    chunks[key] = chunk;
    finest = std::min(finest, chunk->level);
  }

  void clear() {
    for (auto &kv : chunks) delete kv.second;
    chunks.clear();
    finest = max_level;
  }

  void remove(int level, int face, int64_t i, int64_t j) {
    auto it = chunks.find(pack(level, face, i, j));
    if (it == chunks.end()) return;
    delete it->second;
    chunks.erase(it);
  }

  // Face and level-0 index coordinates of a unit direction (dirs_to_face).
  void face_coords(const V3 &d, int &face, double &ka, double &kb) const {
    double ax = std::fabs(d.x), ay = std::fabs(d.y), az = std::fabs(d.z);
    int axis = (ax >= ay && ax >= az) ? 0 : (ay >= az ? 1 : 2);
    double comp = axis == 0 ? d.x : (axis == 1 ? d.y : d.z);
    face = 2 * axis + (comp < 0.0 ? 1 : 0);
    double n = frame_dot(face, 0, d);
    double a = std::atan(frame_dot(face, 1, d) / n) * (4.0 / M_PI);
    double b = std::atan(frame_dot(face, 2, d) / n) * (4.0 / M_PI);
    a = std::min(1.0, std::max(-1.0, a));
    b = std::min(1.0, std::max(-1.0, b));
    ka = (a + 1.0) * (double)n0 / 2.0;
    kb = (b + 1.0) * (double)n0 / 2.0;
  }

  bool contains(const Chunk *c, int face, double ka, double kb) const {
    if (c == nullptr || c->face != face) return false;
    double per = (double)((int64_t)QUADS << c->level);
    return ka >= c->i * per && ka <= (c->i + 1) * per && kb >= c->j * per &&
           kb <= (c->j + 1) * per;
  }

  // The drawn chunk under (face, ka, kb): drawn chunks never overlap, so the
  // first level that has one is the one on screen.
  const Chunk *find(int face, double ka, double kb, const Chunk *hint) const {
    if (contains(hint, face, ka, kb)) return hint;
    for (int level = finest; level <= max_level; ++level) {
      int64_t per = (int64_t)QUADS << level;
      int64_t last = ((int64_t)1 << (max_level - level)) - 1;
      int64_t i = std::min(last, std::max<int64_t>(0, (int64_t)std::floor(ka / per)));
      int64_t j = std::min(last, std::max<int64_t>(0, (int64_t)std::floor(kb / per)));
      auto it = chunks.find(pack(level, face, i, j));
      if (it != chunks.end()) return it->second;
    }
    return nullptr;
  }

  // The drawn chunk under unit direction d, and where in it (quads).
  const Chunk *locate(const V3 &d, const Chunk *hint, double &fu, double &fv) const {
    int face;
    double ka, kb;
    face_coords(d, face, ka, kb);
    const Chunk *c = find(face, ka, kb, hint);
    if (c == nullptr) return nullptr;
    double scale = (double)((int64_t)1 << c->level);
    fu = ka / scale - (double)(c->i * QUADS);
    fv = kb / scale - (double)(c->j * QUADS);
    return c;
  }

  // Radial ray against the drawn triangle under unit direction d
  // (query._triangle_intersect). False where no chunk is drawn.
  bool surface(const V3 &d, Surface &out, const Chunk *hint = nullptr) const {
    double fu, fv;
    const Chunk *c = locate(d, hint, fu, fv);
    if (c == nullptr) return false;
    int col = std::min(QUADS - 1, std::max(0, (int)std::floor(fu)));
    int row = std::min(QUADS - 1, std::max(0, (int)std::floor(fv)));
    int base = row * VERTS + col;
    const float *p = c->pos.data();
    auto vertex = [&](int index) {
      return V3(p[3 * index], p[3 * index + 1], p[3 * index + 2]) + c->centre;
    };
    V3 v00 = vertex(base), v01 = vertex(base + 1), v10 = vertex(base + VERTS),
       v11 = vertex(base + VERTS + 1);
    bool lower = (fu - col) >= (fv - row);
    V3 e1 = lower ? v01 - v00 : v11 - v00;
    V3 e2 = lower ? v11 - v00 : v10 - v00;
    V3 n = e1.cross(e2).normalized();
    double t = n.dot(v00) / n.dot(d);
    out.height = t - radius;
    out.normal = n;
    out.chunk = c;
    if (ocean && out.height < 0.0) {
      out.height = 0.0;
      out.normal = d;
    }
    return true;
  }

  // Highest ground within `reach` metres of the point's radial (sea level
  // counts), or -inf where nothing is drawn: the tile bound when the reach
  // stays within one tile, else the whole chunk's.
  double ceiling(const V3 &p, double reach, const Chunk *&hint) const {
    double fu, fv;
    const Chunk *c = locate(p.normalized(), hint, fu, fv);
    if (c == nullptr) return -INFINITY;
    hint = c;
    double top;
    if (reach <= TILE * spacing_at(c)) {
      int tu = std::min(TILES - 1, std::max(0, (int)(fu / TILE)));
      int tv = std::min(TILES - 1, std::max(0, (int)(fv / TILE)));
      top = c->tile_top[tv][tu];
    } else {
      top = c->h_max;
    }
    return ocean ? std::max(top, 0.0) : top;
  }

  double spacing_at(const Chunk *c) const {
    return spacing0 * (double)((int64_t)1 << (c ? c->level : finest));
  }

  // First crossing of segment a-b into the surface.
  bool segment(const V3 &a, const V3 &b, double &t_hit, V3 &point, V3 &normal) const {
    // Reject: the segment's lowest point over the highest ground of the
    // chunks under its ends and middle.
    const Chunk *hint = nullptr;
    V3 ab = b - a;
    double len2 = ab.dot(ab);
    double reach = 0.5 * std::sqrt(len2);
    double top = std::max(ceiling(a, reach, hint),
                          std::max(ceiling(b, reach, hint), ceiling((a + b) * 0.5, reach, hint)));
    if (top == -INFINITY) return false;
    double s = len2 > 0 ? std::min(1.0, std::max(0.0, -a.dot(ab) / len2)) : 0.0;
    double lowest = (a + ab * s).length();
    if (lowest - radius > top) return false;

    double length = std::sqrt(len2);
    Surface surf;
    auto altitude = [&](double t, bool &ok) {
      V3 p = a + ab * t;
      double r = p.length();
      ok = surface(p * (1.0 / r), surf, hint);
      if (ok) hint = surf.chunk;
      return r - radius - surf.height;
    };
    bool ok;
    double alt0 = altitude(0.0, ok);
    if (ok && alt0 <= 0.0) {
      t_hit = 0.0;
      point = a.normalized() * (radius + surf.height);
      normal = surf.normal;
      return true;
    }
    double step = 0.5 * spacing_at(hint) / std::max(length, 1e-9);
    int steps = std::max(2, (int)std::ceil(1.0 / step));
    double prev = 0.0;
    for (int k = 1; k <= steps; ++k) {
      double t = (double)k / steps;
      double alt = altitude(t, ok);
      if (ok && alt <= 0.0) {
        double lo = prev, hi = t;
        for (int it = 0; it < 24; ++it) {
          double mid = 0.5 * (lo + hi);
          bool mid_ok;
          if (altitude(mid, mid_ok) > 0.0 || !mid_ok) lo = mid; else hi = mid;
        }
        V3 p = a + ab * hi;
        V3 d = p.normalized();
        surface(d, surf, hint);
        t_hit = hi;
        point = d * (radius + surf.height);
        normal = surf.normal;
        return true;
      }
      prev = t;
    }
    return false;
  }

  // Contact of sphere (c, r): the surface point nearest the centre on the
  // triangle plane under it, and that triangle's normal (query.sphere_contact).
  bool sphere(const V3 &c, double r, int samples, V3 &point, V3 &normal, double &distance) const {
    double rc = c.length();
    V3 up = c * (1.0 / rc);
    V3 east = V3(0, 0, 1).cross(up);
    if (east.length() < 1e-6) east = V3(1, 0, 0).cross(up);
    east = east.normalized();
    V3 north = up.cross(east);

    // Reject: below the sphere is clear of the highest ground of the chunks
    // under its centre and four rim points.
    const Chunk *hint = nullptr;
    double top = ceiling(c, r, hint);
    top = std::max(top, ceiling(c + east * r, r, hint));
    top = std::max(top, ceiling(c - east * r, r, hint));
    top = std::max(top, ceiling(c + north * r, r, hint));
    top = std::max(top, ceiling(c - north * r, r, hint));
    if (top == -INFINITY || rc - r - radius > top) return false;

    // Nearest sample of the footprint.
    Surface surf;
    bool found = false;
    double best = INFINITY;
    V3 best_point, best_normal;
    int half = samples / 2;
    double step = half > 0 ? r / half : 0.0;
    for (int y = -half; y <= half; ++y) {
      for (int x = -half; x <= half; ++x) {
        V3 d = (c + east * (x * step) + north * (y * step)).normalized();
        if (!surface(d, surf, hint)) continue;
        hint = surf.chunk;
        V3 p = d * (radius + surf.height);
        V3 diff = c - p;
        double d2 = diff.dot(diff);
        if (d2 < best) {
          best = d2;
          best_point = p;
          best_normal = surf.normal;
          found = true;
        }
      }
    }
    if (!found) return false;

    // Refine: slide to the foot of the centre on the nearest triangle's
    // plane, a few times (converges on smooth ground).
    V3 s = best_point, n = best_normal;
    for (int it = 0; it < 3; ++it) {
      V3 foot = c - n * (c - s).dot(n);
      V3 d = foot.normalized();
      if (!surface(d, surf, hint)) break;
      hint = surf.chunk;
      V3 s2 = d * (radius + surf.height);
      if ((s2 - s).dot(s2 - s) < 1e-6) { s = s2; n = surf.normal; break; }
      s = s2;
      n = surf.normal;
    }
    distance = (c - s).dot(n);
    if (distance < r) {
      point = c - n * distance;
      normal = n;
      return true;
    }
    // The plane under the foot is clear; the nearest sample may not be.
    if (best < r * r) {
      distance = (c - best_point).dot(best_normal);
      if (distance < r) {
        point = c - best_normal * distance;
        normal = best_normal;
        return true;
      }
    }
    return false;
  }

  // Dominant biome at the vertex nearest a point, -1 over the sea.
  int material(const V3 &p) const {
    V3 d = p.normalized();
    Surface surf;
    int face;
    double ka, kb;
    face_coords(d, face, ka, kb);
    const Chunk *c = find(face, ka, kb, nullptr);
    if (c == nullptr) return -2;
    if (ocean && surface(d, surf, c) && surf.height <= 0.0 && surf.normal.dot(d) > 0.999999) {
      return -1;
    }
    double scale = (double)((int64_t)1 << c->level);
    int col = std::min(QUADS, std::max(0, (int)std::lround(ka / scale - (double)(c->i * QUADS))));
    int row = std::min(QUADS, std::max(0, (int)std::lround(kb / scale - (double)(c->j * QUADS))));
    const unsigned char *w = &c->weights[(row * VERTS + col) * BIOMES];
    return (int)(std::max_element(w, w + BIOMES) - w);
  }
};

class CollisionTerrain : public CollisionSolid {
public:
  explicit CollisionTerrain(TerrainData *data) : _data(data), _samples(5) {}
  CollisionTerrain(const CollisionTerrain &copy)
      : CollisionSolid(copy), _data(copy._data), _samples(copy._samples) {}

  virtual CollisionSolid *make_copy() { return new CollisionTerrain(*this); }
  virtual LPoint3 get_collision_origin() const { return LPoint3(0, 0, 0); }
  virtual void output(std::ostream &out) const {
    out << "terrain(" << _data->chunks.size() << " chunks)";
  }

  void set_samples(int samples) { _samples = std::max(1, samples | 1); }

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
    double distance;
    if (!_data->sphere(c, r, _samples, point, normal, distance)) return nullptr;
    PT(CollisionEntry) hit = new CollisionEntry(entry);
    LVector3 n((PN_stdfloat)normal.x, (PN_stdfloat)normal.y, (PN_stdfloat)normal.z);
    hit->set_surface_normal(n);
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
    if (!_data->segment(a, b, t, point, normal)) return nullptr;
    PT(CollisionEntry) hit = new CollisionEntry(entry);
    hit->set_surface_normal(LVector3((PN_stdfloat)normal.x, (PN_stdfloat)normal.y, (PN_stdfloat)normal.z));
    LPoint3 local(point.x - o.x, point.y - o.y, point.z - o.z);
    hit->set_surface_point(local);
    hit->set_interior_point(local);
    return hit;
  }

private:
  PT(TerrainData) _data;
  int _samples;

public:
  static TypeHandle get_class_type() { return _type_handle; }
  static void init_type() {
    CollisionSolid::init_type();
    register_type(_type_handle, "CollisionTerrain", CollisionSolid::get_class_type());
  }
  virtual TypeHandle get_type() const { return get_class_type(); }
  virtual TypeHandle force_init_type() { init_type(); return get_class_type(); }

private:
  static TypeHandle _type_handle;
};

TypeHandle CollisionTerrain::_type_handle;

// ── C API (ctypes) ─────────────────────────────────────────────────────────────

extern "C" {

int wb_init() {
  CollisionTerrain::init_type();
  return (int)CollisionTerrain::get_class_type().get_index();
}

// The Panda3D series the library was built against, major * 100 + minor.
int wb_panda_version() {
  return PANDA_MAJOR_VERSION * 100 + PANDA_MINOR_VERSION;
}

void *wb_terrain_new(double radius, int max_level, long long n0, int ocean) {
  TerrainData *data = new TerrainData(radius, max_level, (int64_t)n0, ocean != 0);
  data->ref();
  return data;
}

void wb_terrain_free(void *handle) {
  unref_delete((TerrainData *)handle);
}

void wb_terrain_set_origin(void *handle, double x, double y, double z) {
  ((TerrainData *)handle)->origin = V3(x, y, z);
}

// vertices: GRID records of `stride` floats each, xyz first (centre-relative);
// weights: GRID * 6 bytes.
void wb_terrain_add_chunk(void *handle, int level, int face, long long i, long long j,
                          const double *centre, const float *vertices, int stride,
                          const float *elevation, float h_min, float h_max,
                          const unsigned char *weights) {
  Chunk *c = new Chunk;
  c->level = level;
  c->face = face;
  c->i = i;
  c->j = j;
  c->centre = V3(centre[0], centre[1], centre[2]);
  c->h_min = h_min;
  c->h_max = h_max;
  c->pos.resize(GRID * 3);
  for (int k = 0; k < GRID; ++k) {
    std::memcpy(&c->pos[3 * k], vertices + (size_t)k * stride, 3 * sizeof(float));
  }
  c->weights.assign(weights, weights + GRID * BIOMES);
  // Tile maxima over each tile's vertices (edges included), then spread to
  // the 8 neighbours; at the chunk's edge the chunk maximum stands in for
  // the neighbour chunk's tiles.
  float raw[TILES][TILES];
  for (int tv = 0; tv < TILES; ++tv) {
    for (int tu = 0; tu < TILES; ++tu) {
      float top = -INFINITY;
      for (int row = tv * TILE; row <= (tv + 1) * TILE; ++row) {
        for (int col = tu * TILE; col <= (tu + 1) * TILE; ++col) {
          top = std::max(top, elevation[row * VERTS + col]);
        }
      }
      raw[tv][tu] = top;
    }
  }
  for (int tv = 0; tv < TILES; ++tv) {
    for (int tu = 0; tu < TILES; ++tu) {
      float top = raw[tv][tu];
      for (int dv = -1; dv <= 1; ++dv) {
        for (int du = -1; du <= 1; ++du) {
          int nv = tv + dv, nu = tu + du;
          if (nv < 0 || nv >= TILES || nu < 0 || nu >= TILES) {
            top = std::max(top, h_max);
          } else {
            top = std::max(top, raw[nv][nu]);
          }
        }
      }
      c->tile_top[tv][tu] = top;
    }
  }
  ((TerrainData *)handle)->add(c);
}

void wb_terrain_remove_chunk(void *handle, int level, int face, long long i, long long j) {
  ((TerrainData *)handle)->remove(level, face, i, j);
}

void wb_terrain_clear(void *handle) {
  ((TerrainData *)handle)->clear();
}

int wb_terrain_num_chunks(void *handle) {
  return (int)((TerrainData *)handle)->chunks.size();
}

// node: the C++ CollisionNode behind a Python panda3d.core.CollisionNode.
int wb_terrain_attach(void *handle, void *node, int samples) {
  CollisionTerrain *solid = new CollisionTerrain((TerrainData *)handle);
  solid->set_samples(samples);
  CollisionNode *collision_node = (CollisionNode *)node;
  collision_node->add_solid(solid);
  return collision_node->get_num_solids();
}

// Direct planet-frame queries (tests, game helpers). out: t, point xyz, normal xyz.
int wb_terrain_segment(void *handle, const double *a, const double *b, double *out) {
  double t;
  V3 point, normal;
  if (!((TerrainData *)handle)->segment(V3(a[0], a[1], a[2]), V3(b[0], b[1], b[2]), t, point, normal)) {
    return 0;
  }
  out[0] = t;
  out[1] = point.x; out[2] = point.y; out[3] = point.z;
  out[4] = normal.x; out[5] = normal.y; out[6] = normal.z;
  return 1;
}

// out: point xyz, normal xyz.
int wb_terrain_sphere(void *handle, const double *c, double r, int samples, double *out) {
  V3 point, normal;
  double distance;
  if (!((TerrainData *)handle)->sphere(V3(c[0], c[1], c[2]), r, samples, point, normal, distance)) {
    return 0;
  }
  out[0] = point.x; out[1] = point.y; out[2] = point.z;
  out[3] = normal.x; out[4] = normal.y; out[5] = normal.z;
  return 1;
}

int wb_terrain_material(void *handle, const double *p) {
  return ((TerrainData *)handle)->material(V3(p[0], p[1], p[2]));
}

}
