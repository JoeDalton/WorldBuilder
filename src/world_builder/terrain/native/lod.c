/* Native LOD selection for world_builder.terrain.quadtree (issue #20).
 *
 * The same pass as LodSelector.select, in double precision, operation for
 * operation: a node splits when the camera is nearer than K times its size,
 * measured to its tangent-frame box; split nodes merge back only past the
 * hysteresis; nodes wholly behind the planet's bulge or beyond the distance
 * cap are culled; a node is replaced by its children only once all of its
 * visible children are ready. The Python version stays as the reference (and
 * the fallback): tests/test_native_lod.py runs both.
 *
 * The chunks' state (height bounds, ready to draw) is pushed from Python as
 * it changes (wb_lod_set_bounds, wb_lod_set_ready), so a pass needs no
 * callback into Python. Nodes live in a growing array, found by key through
 * an open-addressing index; nothing is deleted, except by a rebuild between
 * passes that keeps the nodes with a state once the table grows too large. */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WB_API __declspec(dllexport)
#else
#define WB_API __attribute__((visibility("default")))
#endif

#define QUADS 64
/* The prune threshold: nodes without a state are dropped past it. */
#define NODE_LIMIT 200000
/* quadtree.BEHIND_PENALTY and _BEHIND_COS. */
#define BEHIND_PENALTY 3.0
#define BEHIND_COS (-0.3)
#define PI 3.141592653589793

typedef struct {
  double up[3], t_u[3], t_v[3];
  double u0, u1, v0, v1;
  double cos_radius, sin_radius;
} KeyBox;

typedef struct {
  uint64_t key;
  int32_t level, face, i, j;
  double h_min, h_max; /* valid if has_bounds */
  uint8_t has_bounds, ready, has_box;
  int64_t split_pass; /* the pass that last split it */
  KeyBox box;
  /* The box extents for bounds (gb0, gb1) and its horizon terms, valid while
   * geom_gen is the selector's generation. */
  int64_t geom_gen;
  double gb0, gb1;
  double g[10]; /* u0 u1 v0 v1 w0 w1 cos_r sin_r cos_top sin_top */
} Node;

typedef struct {
  uint64_t key;
  double distance;
  int32_t level;
  int64_t seq;
  int32_t cls;
  double a, b;
} Entry;

typedef struct {
  Entry *items;
  int64_t count, capacity;
} List;

typedef struct {
  /* the grid */
  double radius, spacing0;
  int32_t max_level;
  int64_t n0;
  /* the parameters */
  double split_factor, hysteresis, max_distance, horizon_floor;
  double unknown_b0, unknown_b1; /* a root's heights before it is generated */
  int32_t min_level;
  /* the nodes */
  Node *nodes;
  int64_t count, capacity;
  int64_t *index; /* slot -> node + 1, 0 = empty */
  int64_t slots;  /* a power of two */
  /* per pass */
  int64_t pass, gen;
  double floor;
  double cam[3], dir[3], rho, cos_eye, sin_eye;
  double *limits;
  int64_t seq;
  List draw, wanted, prefetch, keep;
  int32_t *out;
  int64_t out_capacity;
} Lod;

/* ── keys and the index ──────────────────────────────────────────────────── */

static uint64_t pack(int32_t level, int32_t face, int32_t i, int32_t j) {
  return ((uint64_t)level << 42) | ((uint64_t)face << 39) | ((uint64_t)i << 19) |
         (uint64_t)j;
}

static uint64_t mix64(uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebull;
  x ^= x >> 31;
  return x;
}

static void reindex(Lod *lod, int64_t slots) {
  int64_t n, mask = slots - 1;
  free(lod->index);
  lod->index = (int64_t *)calloc((size_t)slots, sizeof(int64_t));
  lod->slots = slots;
  for (n = 0; n < lod->count; n++) {
    int64_t s = (int64_t)(mix64(lod->nodes[n].key) & (uint64_t)mask);
    while (lod->index[s]) s = (s + 1) & mask;
    lod->index[s] = n + 1;
  }
}

/* The node's index, or -1. */
static int64_t find(const Lod *lod, uint64_t key) {
  int64_t mask = lod->slots - 1;
  int64_t s = (int64_t)(mix64(key) & (uint64_t)mask);
  while (lod->index[s]) {
    int64_t n = lod->index[s] - 1;
    if (lod->nodes[n].key == key) return n;
    s = (s + 1) & mask;
  }
  return -1;
}

/* The node's index, made if new. Indices stay valid; pointers into nodes do
 * not (the array grows). */
static int64_t node_of(Lod *lod, int32_t level, int32_t face, int32_t i, int32_t j) {
  uint64_t key = pack(level, face, i, j);
  int64_t n = find(lod, key), mask, s;
  Node *node;
  if (n >= 0) return n;
  if (lod->count == lod->capacity) {
    lod->capacity *= 2;
    lod->nodes = (Node *)realloc(lod->nodes, (size_t)lod->capacity * sizeof(Node));
  }
  if (2 * (lod->count + 1) > lod->slots) reindex(lod, 2 * lod->slots);
  n = lod->count++;
  node = &lod->nodes[n];
  memset(node, 0, sizeof(Node));
  node->key = key;
  node->level = level, node->face = face, node->i = i, node->j = j;
  node->split_pass = -2;
  node->geom_gen = -1;
  mask = lod->slots - 1;
  s = (int64_t)(mix64(key) & (uint64_t)mask);
  while (lod->index[s]) s = (s + 1) & mask;
  lod->index[s] = n + 1;
  return n;
}

/* Between passes: past NODE_LIMIT, keep only the nodes with a state. */
static void prune(Lod *lod) {
  int64_t n, kept = 0;
  if (lod->count <= NODE_LIMIT) return;
  for (n = 0; n < lod->count; n++) {
    Node *node = &lod->nodes[n];
    if (node->has_bounds || node->ready || node->split_pass >= lod->pass - 1)
      lod->nodes[kept++] = *node;
  }
  lod->count = kept;
  reindex(lod, lod->slots);
}

/* ── geometry: cubesphere._point and _key_box ────────────────────────────── */

static const double FRAMES[6][3][3] = {
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},  {{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}},
    {{0, 1, 0}, {-1, 0, 0}, {0, 0, 1}}, {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
    {{0, 0, 1}, {0, 1, 0}, {-1, 0, 0}}, {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}},
};

static void point(int32_t face, double a, double b, double *p) {
  double x = tan(PI / 4.0 * a);
  double y = tan(PI / 4.0 * b);
  const double *n = FRAMES[face][0], *u = FRAMES[face][1], *v = FRAMES[face][2];
  double inv = 1.0 / sqrt(1.0 + x * x + y * y);
  p[0] = (n[0] + x * u[0] + y * v[0]) * inv;
  p[1] = (n[1] + x * u[1] + y * v[1]) * inv;
  p[2] = (n[2] + x * u[2] + y * v[2]) * inv;
}

static void key_box(const Lod *lod, const Node *node, KeyBox *box) {
  double span = 2.0 * (double)((int64_t)QUADS << node->level) / (double)lod->n0;
  double a0 = -1.0 + node->i * span, b0 = -1.0 + node->j * span;
  double angle = (PI / 4.0) * span;
  int steps = (int)ceil(angle / 0.05), k, e, count;
  double boundary[4 * 17][3], up[3], along, norm, step_angle, pad, cos_r;
  double du_lo = INFINITY, du_hi = -INFINITY, dv_lo = INFINITY, dv_hi = -INFINITY;
  const double *u_axis = FRAMES[node->face][1];
  if (steps < 2) steps = 2;
  if (steps > 16) steps = 16;
  count = 0;
  for (e = 0; e < 4; e++) {
    for (k = 0; k <= steps; k++) {
      double f = (double)k / (double)steps;
      double *p = boundary[count++];
      if (e == 0) point(node->face, a0 + span * f, b0, p);
      else if (e == 1) point(node->face, a0 + span, b0 + span * f, p);
      else if (e == 2) point(node->face, a0 + span * f, b0 + span, p);
      else point(node->face, a0, b0 + span * f, p);
    }
  }
  point(node->face, a0 + 0.5 * span, b0 + 0.5 * span, up);
  along = u_axis[0] * up[0] + u_axis[1] * up[1] + u_axis[2] * up[2];
  for (k = 0; k < 3; k++) box->t_u[k] = u_axis[k] - up[k] * along;
  /* Python's ** 2 is C's pow. */
  norm = sqrt(pow(box->t_u[0], 2.0) + pow(box->t_u[1], 2.0) + pow(box->t_u[2], 2.0));
  for (k = 0; k < 3; k++) box->t_u[k] = box->t_u[k] / norm;
  box->t_v[0] = up[1] * box->t_u[2] - up[2] * box->t_u[1];
  box->t_v[1] = up[2] * box->t_u[0] - up[0] * box->t_u[2];
  box->t_v[2] = up[0] * box->t_u[1] - up[1] * box->t_u[0];
  for (k = 0; k < count; k++) {
    const double *p = boundary[k];
    double du = p[0] * box->t_u[0] + p[1] * box->t_u[1] + p[2] * box->t_u[2];
    double dv = p[0] * box->t_v[0] + p[1] * box->t_v[1] + p[2] * box->t_v[2];
    if (du < du_lo) du_lo = du;
    if (du > du_hi) du_hi = du;
    if (dv < dv_lo) dv_lo = dv;
    if (dv > dv_hi) dv_hi = dv;
  }
  step_angle = angle / steps;
  pad = step_angle * step_angle / 8.0 + 1e-9;
  {
    const double *corners[4];
    corners[0] = boundary[0], corners[1] = boundary[steps];
    corners[2] = boundary[2 * steps + 1], corners[3] = boundary[2 * steps + 2];
    cos_r = INFINITY;
    for (k = 0; k < 4; k++) {
      const double *c = corners[k];
      double d = c[0] * up[0] + c[1] * up[1] + c[2] * up[2];
      if (d < cos_r) cos_r = d;
    }
  }
  memcpy(box->up, up, sizeof(up));
  box->u0 = du_lo - pad, box->u1 = du_hi + pad;
  box->v0 = dv_lo - pad, box->v1 = dv_hi + pad;
  box->cos_radius = cos_r;
  {
    double s = 1.0 - cos_r * cos_r;
    box->sin_radius = sqrt(s > 0.0 ? s : 0.0);
  }
}

/* ── the pass: LodSelector._measure and _visit ───────────────────────────── */

/* Distance to the node's box for height bounds (b0, b1); *culled set if it is
 * wholly beyond the horizon or the distance cap. */
static double measure(Lod *lod, int64_t n, double b0, double b1, int *culled) {
  Node *node = &lod->nodes[n];
  const double *g, *t_u, *t_v, *w;
  double p, d, total, distance, cos_phi, s, cos_alpha, cos_beta;
  double x = lod->cam[0], y = lod->cam[1], z = lod->cam[2];
  if (!node->has_box) {
    key_box(lod, node, &node->box);
    node->has_box = 1;
  }
  if (node->geom_gen != lod->gen || node->gb0 != b0 || node->gb1 != b1) {
    const KeyBox *kb = &node->box;
    double r0 = lod->radius + b0, r1 = lod->radius + b1, cos_top;
    double *e = node->g;
    e[0] = fmin(kb->u0 * r0, kb->u0 * r1), e[1] = fmax(kb->u1 * r0, kb->u1 * r1);
    e[2] = fmin(kb->v0 * r0, kb->v0 * r1), e[3] = fmax(kb->v1 * r0, kb->v1 * r1);
    e[4] = kb->cos_radius * r0, e[5] = r1;
    e[6] = kb->cos_radius, e[7] = kb->sin_radius;
    cos_top = r1 > lod->floor ? lod->floor / r1 : 1.0;
    e[8] = cos_top;
    e[9] = sqrt(fmax(0.0, 1.0 - cos_top * cos_top));
    node->gb0 = b0, node->gb1 = b1, node->geom_gen = lod->gen;
  }
  g = node->g, t_u = node->box.t_u, t_v = node->box.t_v, w = node->box.up;
  p = t_u[0] * x + t_u[1] * y + t_u[2] * z;
  d = p < g[0] ? g[0] - p : (p > g[1] ? p - g[1] : 0.0);
  total = d * d;
  p = t_v[0] * x + t_v[1] * y + t_v[2] * z;
  d = p < g[2] ? g[2] - p : (p > g[3] ? p - g[3] : 0.0);
  total += d * d;
  p = w[0] * x + w[1] * y + w[2] * z;
  d = p < g[4] ? g[4] - p : (p > g[5] ? p - g[5] : 0.0);
  distance = sqrt(total + d * d);

  *culled = 0;
  if (lod->max_distance >= 0.0 && distance > lod->max_distance) {
    *culled = 1;
    return distance;
  }
  cos_phi = lod->dir[0] * w[0] + lod->dir[1] * w[1] + lod->dir[2] * w[2];
  if (cos_phi >= g[6]) return distance;
  s = 1.0 - cos_phi * cos_phi;
  cos_alpha = cos_phi * g[6] + (s > 0.0 ? sqrt(s) : 0.0) * g[7];
  cos_beta = lod->cos_eye * g[8] - lod->sin_eye * g[9];
  *culled = cos_alpha < cos_beta;
  return distance;
}

static void push(List *list, uint64_t key, int32_t level, double distance, int64_t seq) {
  if (list->count == list->capacity) {
    list->capacity = list->capacity ? 2 * list->capacity : 1024;
    list->items = (Entry *)realloc(list->items, (size_t)list->capacity * sizeof(Entry));
  }
  list->items[list->count].key = key;
  list->items[list->count].level = level;
  list->items[list->count].distance = distance;
  list->items[list->count].seq = seq;
  list->count++;
}

static void bounds_of(const Node *node, double fb0, double fb1, double *b0, double *b1) {
  if (node->has_bounds) {
    *b0 = node->h_min, *b1 = node->h_max;
  } else {
    *b0 = fb0, *b1 = fb1;
  }
}

static void visit(Lod *lod, int64_t n, double b0, double b1, double distance) {
  Node *node = &lod->nodes[n];
  int32_t level = node->level, face = node->face, i = node->i, j = node->j;
  uint64_t key = node->key;
  double limit = lod->limits[level];
  if (node->split_pass == lod->pass - 1) limit *= lod->hysteresis;
  if (level > lod->min_level && distance < limit) {
    int64_t child[4];
    double cb0[4], cb1[4], cd[4];
    int visible[4], missing = 0, c, culled;
    node->split_pass = lod->pass;
    for (c = 0; c < 4; c++) {
      child[c] = node_of(lod, level - 1, face, 2 * i + (c & 1), 2 * j + (c >> 1));
      push(&lod->keep, lod->nodes[child[c]].key, level - 1, 0.0, 0);
    }
    for (c = 0; c < 4; c++) {
      const Node *ch = &lod->nodes[child[c]];
      bounds_of(ch, b0, b1, &cb0[c], &cb1[c]);
      cd[c] = measure(lod, child[c], cb0[c], cb1[c], &culled);
      ch = &lod->nodes[child[c]];
      visible[c] = !culled;
      if (culled) {
        if (!ch->ready) push(&lod->prefetch, ch->key, ch->level, cd[c], lod->seq++);
        continue;
      }
      if (!missing && !ch->ready) missing = 1;
    }
    if (!missing) {
      for (c = 0; c < 4; c++)
        if (visible[c]) visit(lod, child[c], cb0[c], cb1[c], cd[c]);
      return;
    }
    for (c = 0; c < 4; c++) {
      const Node *ch = &lod->nodes[child[c]];
      if (visible[c] && !ch->ready)
        push(&lod->wanted, ch->key, ch->level, cd[c], lod->seq++);
    }
  }
  node = &lod->nodes[n];
  if (node->ready) {
    /* Already kept: every visited node is a root or a split node's child. */
    push(&lod->draw, key, level, distance, 0);
  } else {
    push(&lod->wanted, key, level, distance, lod->seq++);
  }
}

static int by_rank(const void *pa, const void *pb) {
  const Entry *a = (const Entry *)pa, *b = (const Entry *)pb;
  if (a->cls != b->cls) return a->cls < b->cls ? -1 : 1;
  if (a->a != b->a) return a->a < b->a ? -1 : 1;
  if (a->b != b->b) return a->b < b->b ? -1 : 1;
  return a->seq < b->seq ? -1 : (a->seq > b->seq);
}

/* LodSelector._priority: ground nothing covers first (coarse first), then
 * the largest on screen, those behind the view as if farther. */
static void rank_wanted(Lod *lod, const double *forward) {
  double fx = 0, fy = 0, fz = 0;
  int64_t k;
  if (forward) {
    double norm;
    fx = forward[0], fy = forward[1], fz = forward[2];
    norm = sqrt(fx * fx + fy * fy + fz * fz);
    if (norm == 0.0) norm = 1.0;
    fx = fx / norm, fy = fy / norm, fz = fz / norm;
  }
  for (k = 0; k < lod->wanted.count; k++) {
    Entry *e = &lod->wanted.items[k];
    int64_t n = find(lod, e->key), parent;
    const Node *node = &lod->nodes[n];
    parent = e->level == lod->max_level
                 ? -1
                 : find(lod, pack(e->level + 1, node->face, node->i >> 1, node->j >> 1));
    if (e->level == lod->max_level || parent < 0 || !lod->nodes[parent].ready) {
      e->cls = 0, e->a = -(double)e->level, e->b = e->distance;
    } else {
      double score = e->distance / (lod->spacing0 * (double)((int64_t)1 << e->level) * QUADS);
      if (forward && score > 1.0) {
        const double *u = node->box.up;
        double dx = u[0] * lod->radius - lod->cam[0];
        double dy = u[1] * lod->radius - lod->cam[1];
        double dz = u[2] * lod->radius - lod->cam[2];
        double along = dx * fx + dy * fy + dz * fz;
        if (along < BEHIND_COS * sqrt(dx * dx + dy * dy + dz * dz)) score *= BEHIND_PENALTY;
      }
      e->cls = 1, e->a = score, e->b = 0.0;
    }
  }
  qsort(lod->wanted.items, (size_t)lod->wanted.count, sizeof(Entry), by_rank);
}

/* ── the API ─────────────────────────────────────────────────────────────── */

WB_API Lod *wb_lod_new(double radius, int32_t max_level, int64_t n0, double spacing0) {
  Lod *lod = (Lod *)calloc(1, sizeof(Lod));
  lod->radius = radius, lod->max_level = max_level, lod->n0 = n0, lod->spacing0 = spacing0;
  lod->split_factor = 2.0, lod->hysteresis = 1.15, lod->max_distance = -1.0;
  lod->capacity = 4096;
  lod->nodes = (Node *)malloc((size_t)lod->capacity * sizeof(Node));
  reindex(lod, 8192);
  lod->limits = (double *)calloc((size_t)max_level + 1, sizeof(double));
  lod->pass = 2;
  lod->floor = NAN;
  return lod;
}

WB_API void wb_lod_free(Lod *lod) {
  if (!lod) return;
  free(lod->nodes), free(lod->index), free(lod->limits), free(lod->out);
  free(lod->draw.items), free(lod->wanted.items), free(lod->prefetch.items),
      free(lod->keep.items);
  free(lod);
}

/* max_distance < 0: no cap. unknown_b0, unknown_b1: the heights a root is
 * assumed to span before it is generated (the body's height bounds). */
WB_API void wb_lod_set_params(Lod *lod, double split_factor, double hysteresis,
                              double max_distance, double horizon_floor, int32_t min_level,
                              double unknown_b0, double unknown_b1) {
  lod->split_factor = split_factor, lod->hysteresis = hysteresis;
  lod->max_distance = max_distance, lod->horizon_floor = horizon_floor;
  lod->min_level = min_level;
  if (unknown_b0 != lod->unknown_b0 || unknown_b1 != lod->unknown_b1) lod->gen += 1;
  lod->unknown_b0 = unknown_b0, lod->unknown_b1 = unknown_b1;
}

/* Forget the hysteresis state. */
WB_API void wb_lod_reset(Lod *lod) { lod->pass += 2; }

/* keys: (n, 4) level, face, i, j; bounds: (n, 2) h_min, h_max, or NULL to
 * forget them. */
WB_API void wb_lod_set_bounds(Lod *lod, const int32_t *keys, const double *bounds, int64_t n) {
  int64_t k;
  for (k = 0; k < n; k++) {
    const int32_t *q = keys + 4 * k;
    int64_t node = node_of(lod, q[0], q[1], q[2], q[3]);
    lod->nodes[node].has_bounds = bounds != NULL;
    if (bounds) {
      lod->nodes[node].h_min = bounds[2 * k];
      lod->nodes[node].h_max = bounds[2 * k + 1];
    }
  }
}

WB_API void wb_lod_set_ready(Lod *lod, const int32_t *keys, int32_t ready, int64_t n) {
  int64_t k;
  for (k = 0; k < n; k++) {
    const int32_t *q = keys + 4 * k;
    lod->nodes[node_of(lod, q[0], q[1], q[2], q[3])].ready = (uint8_t)(ready != 0);
  }
}

/* One pass. camera, forward: (3,) planet frame; forward may be NULL. The
 * results: wb_lod_result. */
WB_API void wb_lod_select(Lod *lod, const double *camera, const double *forward) {
  double x = camera[0], y = camera[1], z = camera[2], floor, rho;
  int32_t level, face;
  prune(lod);
  lod->pass += 1;
  lod->draw.count = lod->wanted.count = lod->prefetch.count = lod->keep.count = 0;
  lod->seq = 0;
  lod->cam[0] = x, lod->cam[1] = y, lod->cam[2] = z;
  rho = sqrt(x * x + y * y + z * z);
  lod->rho = rho;
  if (rho > 0.0) {
    lod->dir[0] = x / rho, lod->dir[1] = y / rho, lod->dir[2] = z / rho;
  } else {
    lod->dir[0] = 0.0, lod->dir[1] = 0.0, lod->dir[2] = 1.0;
  }
  floor = lod->radius + lod->horizon_floor;
  if (floor != lod->floor) lod->gen += 1; /* the horizon terms depend on it */
  lod->floor = floor;
  lod->cos_eye = rho > floor ? floor / rho : 1.0;
  lod->sin_eye = sqrt(fmax(0.0, 1.0 - lod->cos_eye * lod->cos_eye));
  for (level = 0; level <= lod->max_level; level++)
    lod->limits[level] =
        lod->split_factor * (lod->spacing0 * (double)((int64_t)1 << level) * QUADS);
  for (face = 0; face < 6; face++) {
    int64_t n = node_of(lod, lod->max_level, face, 0, 0);
    double b0 = lod->unknown_b0, b1 = lod->unknown_b1, distance;
    int culled;
    push(&lod->keep, lod->nodes[n].key, lod->max_level, 0.0, 0);
    if (lod->nodes[n].has_bounds) b0 = lod->nodes[n].h_min, b1 = lod->nodes[n].h_max;
    distance = measure(lod, n, b0, b1, &culled);
    if (!culled) visit(lod, n, b0, b1, distance);
  }
  rank_wanted(lod, forward);
  {
    int64_t k;
    for (k = 0; k < lod->prefetch.count; k++) {
      Entry *e = &lod->prefetch.items[k];
      e->cls = 0, e->a = e->distance, e->b = 0.0;
    }
    qsort(lod->prefetch.items, (size_t)lod->prefetch.count, sizeof(Entry), by_rank);
  }
}

/* The last pass's keys, (count, 4) int32, valid until the next call: which =
 * 0 draw, 1 wanted (most urgent first), 2 prefetch, 3 keep. */
WB_API const int32_t *wb_lod_result(Lod *lod, int32_t which, int64_t *count) {
  const List *list = which == 0   ? &lod->draw
                     : which == 1 ? &lod->wanted
                     : which == 2 ? &lod->prefetch
                                  : &lod->keep;
  int64_t k;
  if (list->count * 4 > lod->out_capacity) {
    lod->out_capacity = list->count * 4 + 1024;
    lod->out = (int32_t *)realloc(lod->out, (size_t)lod->out_capacity * sizeof(int32_t));
  }
  for (k = 0; k < list->count; k++) {
    uint64_t key = list->items[k].key;
    lod->out[4 * k] = (int32_t)(key >> 42);
    lod->out[4 * k + 1] = (int32_t)((key >> 39) & 7u);
    lod->out[4 * k + 2] = (int32_t)((key >> 19) & 0xfffffu);
    lod->out[4 * k + 3] = (int32_t)(key & 0x7ffffu);
  }
  *count = list->count;
  return lod->out;
}
