#version 330

// Terrain chunk vertex shader: the true sphere (chunks are already curved), the
// sea-surface mode seen from high up, coastline depth separation, and the clip
// distance the ocean's reflection pass needs.

in vec4 p3d_Vertex;     // relative to the chunk centre, planet axes (metres)
in vec3 p3d_Normal;
in float height;        // metres above sea level (a skirt's own, below its edge)
in vec4 biome_a;        // snow, rock, grass, forest
in vec4 biome_b;        // desert, beach, -, skirt flag

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;   // chunk -> render; a translation (rotation-safe)
uniform vec4 p3d_ClipPlane[1];  // reflection pass only (view space)

uniform vec3  uCamPos;          // MAIN camera in the render frame (also in the mirror pass)
uniform vec3  uPlanetCentre;    // planet centre in the render frame
uniform int   uSeaSurface;      // 1: lift the seabed to sea level, shade it as water
uniform float uDepthBias;       // coastline separation, in depth-buffer steps
uniform vec4  uChunk;           // first vertex's uv (wrapped), uv metres per vertex
                                // step, skirt depth
uniform int   uFace;            // cube face, for the texture tangents

out vec3 vRelPos;     // position relative to the camera, render frame
out vec2 vUv;         // ground-anchored texture coordinate, metres
out float vHeight;    // height above sea level as drawn, metres
out float vSeabed;    // ground height before the sea-surface lift, metres
out vec3 vNormal;
out vec3 vUp;         // local vertical
out vec4 vBiomeA;
out vec4 vBiomeB;
flat out int vFace;

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

void main() {
    vec4 local = p3d_Vertex;
    mat3 rotation = mat3(p3d_ModelMatrix);
    vec3 world = (p3d_ModelMatrix * local).xyz;
    vec3 up = normalize(world - uPlanetCentre);
    vec3 N = normalize(rotation * p3d_Normal);
    bool skirt = biome_b.a > 0.5;
    float ground = skirt ? height + uChunk.w : height;  // the (edge) ground height
    float drawn = height;

    if (uSeaSurface == 1 && ground < 0.0) {
        // Up to sea level; a skirt keeps hanging its depth below it.
        vec3 lift = up * -ground;
        local.xyz += transpose(rotation) * lift;
        world += lift;
        drawn -= ground;
        N = up;
    }

    gl_Position = p3d_ModelViewProjectionMatrix * local;

    // Land just above the sea and seabed just below it sit within one depth
    // step of the ocean at range; nudging land nearer and seabed farther by a
    // fixed number of steps keeps the coastline stable at every distance.
    // Not needed when the terrain draws the sea itself.
    float side = uSeaSurface == 1 ? 0.0 : clamp(drawn, -1.0, 1.0);
    gl_Position.z -= side * uDepthBias * (2.0 / 16777216.0) * gl_Position.w;

    gl_ClipDistance[0] = dot(p3d_ClipPlane[0], p3d_ModelViewMatrix * local);

    vRelPos = world - uCamPos;
    vUv = uChunk.xy + gridCoord(gl_VertexID) * uChunk.z;
    vHeight = drawn;
    vSeabed = ground;
    vNormal = N;
    vUp = up;
    vBiomeA = biome_a;
    vBiomeB = biome_b;
    vFace = uFace;
}
