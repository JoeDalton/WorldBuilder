// Where the trees and bushes grow: one plant at most per cell of a square
// grid on each cube face, in its texture coordinates (vUv, metres), chosen by
// an integer hash of the cell and the biome weights there. Shared by the
// canopy's march (canopy.frag) and the ground under it (vegetation_far.glsl);
// world_builder.terrain.vegetation.plants is its twin in numpy, bit for bit
// on the hash (the tests compare them, and read these constants).
// Include once per shader with #pragma include.

// The cell's side, uv metres: UV_PERIOD (4096) holds a whole number of them,
// so cells match across the wrap of the texture coordinates.
const float VEG_CELL = 6.4;
const int VEG_CELLS_PER_PERIOD = 640;

// The kinds of plant.
const int VEG_NONE = 0;
const int VEG_TREE = 1;
const int VEG_BUSH = 2;
const int VEG_DRY = 3;   // a dry, scorched desert bush

// Occupancy per biome: the share of cells holding each kind, all biome.
const float VEG_TREE_FOREST = 0.85;
const float VEG_TREE_GRASS = 0.04;
const float VEG_BUSH_GRASS = 0.20;
const float VEG_DRY_DESERT = 0.025;
const float VEG_TREE_ROCK = 0.01;
const float VEG_BUSH_ROCK = 0.03;

// Sizes, metres: height and crown radius ranges per kind.
const vec2 VEG_TREE_HEIGHT = vec2(9.0, 21.0);
const vec2 VEG_TREE_RADIUS = vec2(2.0, 3.0);
const vec2 VEG_BUSH_HEIGHT = vec2(0.8, 2.0);
const vec2 VEG_BUSH_RADIUS = vec2(0.6, 1.4);
const vec2 VEG_DRY_HEIGHT = vec2(0.5, 1.2);
const vec2 VEG_DRY_RADIUS = vec2(0.5, 1.0);
// A crown may reach this far past its cell's centre, in cells: a little
// into its neighbours (the march tests a cell's own plant, then the next
// cell's on a hit).
const float VEG_REACH = 0.5;
// The tallest plant above its ground, with its crown's noise: the shell's
// height over trees and bushes (canopy.vert), metres.
const float VEG_HULL_TREE = 23.0;
const float VEG_HULL_BUSH = 2.4;

// A tree's crown: its vertical semi-axis a share of its height, within
// these many crown radii (round to oval), its top at the tree's height (a
// trunk below). A bush's: centre and vertical semi-axis as shares of its
// height (it sits a little sunk in the ground).
const float VEG_TREE_CROWN = 0.32;
const vec2 VEG_TREE_CROWN_ASPECT = vec2(1.2, 2.0);
const vec2 VEG_BUSH_CROWN = vec2(0.42, 0.58);

uniform int uVegSeed;  // the planet's seed: each planet its own plants

uvec3 vegHash(uvec3 v) {
    // pcg3d (Jarzynski and Olano, 2020)
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

struct Plant {
    int kind;        // VEG_NONE: an empty cell
    vec2 uv;         // its centre, uv metres (the cell's own frame: see plantAt)
    float height;    // metres
    float radius;    // its crown's, metres
    float tint;      // in [0, 1): its colour's variation
};

// The plant of a cell (cell: its index in the chunk's period, the period's
// index: uv = cell * VEG_CELL there), on face *face*, where the biome
// weights are forest, grass, desert and rock.
Plant plantAt(ivec2 cell, ivec2 period, int face, vec4 weights) {
    ivec2 id = cell + period * VEG_CELLS_PER_PERIOD;
    uvec3 h1 = vegHash(uvec3(uint(id.x), uint(id.y), uint(face) + 8u * uint(uVegSeed)));
    vec3 r1 = vec3(h1 >> 8u) / 16777216.0;
    float pTree = VEG_TREE_FOREST * weights.x + VEG_TREE_GRASS * weights.y
                + VEG_TREE_ROCK * weights.w;
    float pBush = VEG_BUSH_GRASS * weights.y + VEG_BUSH_ROCK * weights.w;
    float pDry = VEG_DRY_DESERT * weights.z;
    Plant p;
    p.kind = r1.x < pTree ? VEG_TREE
           : r1.x < pTree + pBush ? VEG_BUSH
           : r1.x < pTree + pBush + pDry ? VEG_DRY : VEG_NONE;
    if (p.kind == VEG_NONE) {
        return p;  // most cells of grassland and desert: nothing more to draw
    }
    uvec3 h2 = vegHash(h1 ^ uvec3(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u));
    vec3 r2 = vec3(h2 >> 8u) / 16777216.0;
    vec2 height = p.kind == VEG_TREE ? VEG_TREE_HEIGHT
                : p.kind == VEG_BUSH ? VEG_BUSH_HEIGHT : VEG_DRY_HEIGHT;
    vec2 radius = p.kind == VEG_TREE ? VEG_TREE_RADIUS
                : p.kind == VEG_BUSH ? VEG_BUSH_RADIUS : VEG_DRY_RADIUS;
    p.height = mix(height.x, height.y, r1.y);
    p.radius = mix(radius.x, radius.y, r1.z);
    // Off the cell's centre as far as its crown stays within VEG_REACH.
    float play = max(VEG_REACH * VEG_CELL - p.radius, 0.0);
    p.uv = (vec2(cell) + 0.5) * VEG_CELL + (2.0 * r2.xy - 1.0) * play;
    p.tint = r2.z;
    return p;
}

// A plant's crown: its centre above the base and its vertical semi-axis,
// metres.
vec2 plantCrown(Plant p) {
    if (p.kind == VEG_TREE) {
        float b = clamp(VEG_TREE_CROWN * p.height, VEG_TREE_CROWN_ASPECT.x * p.radius,
                        VEG_TREE_CROWN_ASPECT.y * p.radius);
        return vec2(p.height - b, b);
    }
    return p.height * VEG_BUSH_CROWN;
}

// The mean share of the ground the crowns cover, seen from straight above
// (overlaps counted twice), and how much of that is trees, where the weights
// are forest, grass, desert and rock: the far look's coverage.
// E[pi r^2] / VEG_CELL^2 per kind, r uniform in its range.
float vegCrownArea(vec2 r) {
    return 3.14159265 * (r.y * r.y * r.y - r.x * r.x * r.x) / (3.0 * (r.y - r.x))
           / (VEG_CELL * VEG_CELL);
}

vec2 vegCoverage(vec4 weights) {
    float trees = (VEG_TREE_FOREST * weights.x + VEG_TREE_GRASS * weights.y
                   + VEG_TREE_ROCK * weights.w) * vegCrownArea(VEG_TREE_RADIUS);
    float bushes = (VEG_BUSH_GRASS * weights.y + VEG_BUSH_ROCK * weights.w)
                       * vegCrownArea(VEG_BUSH_RADIUS)
                 + VEG_DRY_DESERT * weights.z * vegCrownArea(VEG_DRY_RADIUS);
    return vec2(trees + bushes, trees);
}

// The colours (linear): a tree's foliage, a bush's, a dry bush's, each by
// its tint; the trees of a forest a little darker than lone ones.
vec3 plantColor(Plant p, float forest) {
    if (p.kind == VEG_TREE) {
        vec3 a = vec3(0.040, 0.075, 0.022), b = vec3(0.060, 0.090, 0.030);
        return mix(a, b, p.tint) * mix(1.15, 0.85, forest);
    }
    if (p.kind == VEG_BUSH) {
        return mix(vec3(0.060, 0.075, 0.030), vec3(0.085, 0.085, 0.040), p.tint);
    }
    return mix(vec3(0.060, 0.045, 0.030), vec3(0.030, 0.026, 0.022), p.tint);
}

// The far look's canopy colour: the trees' and bushes' mean, by their share
// of the cover.
vec3 canopyMeanColor(vec4 weights) {
    vec2 cover = vegCoverage(weights);
    float bushes = max(cover.x - cover.y, 0.0);
    vec3 trees = mix(vec3(0.040, 0.075, 0.022), vec3(0.060, 0.090, 0.030), 0.5)
               * mix(1.15, 0.85, weights.x);
    float dry = VEG_DRY_DESERT * weights.z * vegCrownArea(VEG_DRY_RADIUS);
    vec3 bush = mix(vec3(0.0725, 0.080, 0.035), vec3(0.045, 0.0355, 0.026),
                    bushes > 0.0 ? clamp(dry / bushes, 0.0, 1.0) : 0.0);
    return cover.x > 0.0 ? (trees * cover.y + bush * bushes) / cover.x : trees;
}
