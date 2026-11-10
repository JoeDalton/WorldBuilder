// A chunk's vertex grid (cubesphere.QUADS quads a side), shared by the
// terrain's vertex shaders (terrain.vert, canopy.vert). Include once per
// shader with #pragma include.

const int QUADS = 64;
const int VERTS = QUADS + 1;

// (column, row) of a vertex in the chunk grid; skirt vertices repeat their edge
// vertex, in chunkgen.skirt_ring's order.
vec2 gridCoord(int id) {
    if (id < VERTS * VERTS) {
        return vec2(id % VERTS, id / VERTS);
    }
    int p = id - VERTS * VERTS;
    int side = p / QUADS;
    float t = float(p % QUADS);
    float n = float(QUADS);
    if (side == 0) return vec2(t, 0.0);       // south
    if (side == 1) return vec2(n, t);         // east
    if (side == 2) return vec2(n - t, n);     // north
    return vec2(0.0, n - t);                  // west
}
