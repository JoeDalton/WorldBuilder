#version 330

// Terrain chunk vertex shader: the true sphere (chunks are already curved), the
// sea surface (seabed vertices lifted to the water, which the fragment shader
// shades as sea), and the clip distance the sea's reflection pass needs.

in vec4 p3d_Vertex;     // relative to the chunk centre, planet axes (metres)
in vec3 p3d_Normal;
in float height;        // metres above sea level (a skirt's own, below its edge)
in vec4 biome_a;        // snow, rock, grass, forest
in vec4 biome_b;        // desert, the strata mask (beach is the rest of the
                        // weights), then the chunk's slot (low byte; high 7
                        // bits) with the skirt flag (top bit), set at upload

#pragma include "../../ocean/shaders/swell.glsl"
#pragma include "cube_faces.glsl"

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;   // chunk -> render: the body's orientation, then a translation
uniform vec4 p3d_ClipPlane[1];  // reflection pass only (view space)

uniform vec3  uCamPos;          // MAIN camera in the render frame (also in the mirror pass)
uniform vec3  uPlanetCentre;    // planet centre in the render frame
uniform int   uSeaSurface;      // 1: the planet has a sea: the seabed is lifted
                                // to it and shaded as water
// Each chunk's own values, by its slot (Terrain's chunk table; one texture
// for all chunks rather than inputs per chunk, which made Panda set every
// chunk's state apart: ~2.5 ms a frame). Per slot, three texels:
//   0: the chunk's centre, planet frame (the swell's point), and
//      1 / (4 vertex spacing): the finest swell it draws;
//   1: its first vertex's uv (wrapped), uv metres per vertex step, skirt depth;
//   2: its level and cube face.
uniform sampler2D uChunkTable;

out vec3 vRelPos;     // position relative to the camera, render frame
out vec2 vUv;         // ground-anchored texture coordinate, metres
out float vHeight;    // height above sea level as drawn, metres
out float vSeabed;    // ground height before the sea-surface lift, metres
out vec3 vNormal;
out vec3 vUp;         // local vertical
out vec4 vBiomeA;
out vec4 vBiomeB;
out vec4 vSwell;     // the swell: gradient per metre (planet axes), field
flat out int vFace;
flat out vec3 vFaceU;  // the face's texture axes, render frame (the body turns)
flat out vec3 vFaceV;
flat out int vLevel;
flat out float vStep;  // uv metres per vertex step (about the vertex spacing)

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
    int high = int(biome_b.a * 255.0 + 0.5);
    int slot = int(biome_b.b * 255.0 + 0.5) + 256 * (high & 127);
    bool skirt = high >= 128;
    vec4 uChunkCentre = texelFetch(uChunkTable, ivec2(slot, 0), 0);
    vec4 uChunk = texelFetch(uChunkTable, ivec2(slot, 1), 0);
    vec4 ids = texelFetch(uChunkTable, ivec2(slot, 2), 0);
    float ground = skirt ? height + uChunk.w : height;  // the (edge) ground height
    float drawn = height;
    vSwell = vec4(0.0, 0.0, 0.0, SWELL_MEAN);

    if (uSeaSurface == 1 && ground < 0.0) {
        // Up to the water: sea level and the swell, which has as much detail
        // as the chunk's vertices can carry. A skirt keeps hanging its depth
        // below its edge's water.
        vec3 planet = uChunkCentre.xyz + p3d_Vertex.xyz;
        if (skirt) {
            planet += up * uChunk.w;  // its edge's point
        }
        // The field and its gradient, for the near look's shading too: the
        // geometry's own swell, so the normal follows the drawn water.
        vec3 grad;
        float field = swellField(planet - up * ground, uChunkCentre.w, grad);
        float water = uSwellAmplitude * swellDepthFactor(-ground) * (field - SWELL_MEAN);
        vSwell = vec4(grad, field);
        vec3 lift = up * (water - ground);
        local.xyz += transpose(rotation) * lift;
        world += lift;
        drawn += water - ground;
        N = up;
    }

    gl_Position = p3d_ModelViewProjectionMatrix * local;

    // The sea's reflection camera clips at the water plane (a clip plane on
    // its initial state), which a shader must apply itself. Panda enables the
    // clip distance only while a clip plane is on, and passes a zero plane
    // otherwise. The sea is not in its own reflection: in that pass its
    // vertices are clipped too (clipping, not a discard, which would cost the
    // early depth test everywhere).
    bool mirror = p3d_ClipPlane[0] != vec4(0.0);
    gl_ClipDistance[0] = mirror && uSeaSurface == 1 && ground < 0.0
        ? -1.0 : dot(p3d_ClipPlane[0], p3d_ModelViewMatrix * local);

    vRelPos = world - uCamPos;
    vUv = uChunk.xy + gridCoord(gl_VertexID) * uChunk.z;
    vHeight = drawn;
    vSeabed = ground;
    vNormal = N;
    vUp = up;
    vBiomeA = biome_a;
    vBiomeB = biome_b;
    vFace = int(ids.y + 0.5);
    vFaceU = rotation * FACE_U[vFace];
    vFaceV = rotation * FACE_V[vFace];
    vLevel = int(ids.x + 0.5);
    vStep = uChunk.z;
}
