// The seabed under the water and what the swell takes from it. Included by
// ocean.vert and ocean.frag; the textures and mappings are set by
// world_builder.ocean.seabed.SeabedMap.

// Cascades baked, finest first (0 = no seabed: deep sea everywhere).
uniform int       uSeabedCount;
uniform sampler2D uSeabedTex0;
uniform sampler2D uSeabedTex1;
uniform sampler2D uSeabedTex2;
// The ocean plane's (x, y, 1) to each cascade's texture u and v.
uniform vec3      uSeabedU[3];
uniform vec3      uSeabedV[3];
// Each cascade's texel, metres: how finely the depth it gives is placed.
uniform float     uSeabedTexel[3];

// seabed.py's DEEP_SEA_M: what no cascade covers.
#define SEABED_DEEP   -10000.0
// Each cascade hands over to the next coarser across this fraction of its
// side, so no edge shows.
#define SEABED_EDGE   0.1

// How much of a cascade covers uv: 1 inside, falling to 0 at its edges.
float seabedCover(vec2 uv) {
    vec2 d = min(uv, 1.0 - uv);
    return clamp(min(d.x, d.y) / SEABED_EDGE, 0.0, 1.0);
}

// Height of the sea floor under the plane's xy, metres (negative under
// water), and the texel it was read at (blended like it; 1e9 where no
// cascade reaches). Mostly one texture read: the finest cascade covering xy,
// blended into the next only in its edge band.
float seabedHeight(vec2 xy, out float texel) {
    vec3  p    = vec3(xy, 1.0);
    float h    = 0.0;
    float rest = 1.0;  // what the finer cascades left to cover
    texel = 0.0;
    if (uSeabedCount > 0) {
        vec2  uv = vec2(dot(uSeabedU[0], p), dot(uSeabedV[0], p));
        float w  = seabedCover(uv);
        if (w > 0.0) {
            h += w * textureLod(uSeabedTex0, uv, 0.0).r;
            texel += w * uSeabedTexel[0];
            rest -= w;
        }
    }
    if (uSeabedCount > 1 && rest > 0.0) {
        vec2  uv = vec2(dot(uSeabedU[1], p), dot(uSeabedV[1], p));
        float w  = rest * seabedCover(uv);
        if (w > 0.0) {
            h += w * textureLod(uSeabedTex1, uv, 0.0).r;
            texel += w * uSeabedTexel[1];
            rest -= w;
        }
    }
    if (uSeabedCount > 2 && rest > 0.0) {
        vec2  uv = vec2(dot(uSeabedU[2], p), dot(uSeabedV[2], p));
        float w  = rest * seabedCover(uv);
        if (w > 0.0) {
            h += w * textureLod(uSeabedTex2, uv, 0.0).r;
            texel += w * uSeabedTexel[2];
            rest -= w;
        }
    }
    texel += rest * 1e9;
    return h + rest * SEABED_DEEP;
}

float seabedHeight(vec2 xy) {
    float texel;
    return seabedHeight(xy, texel);
}

// The swell's share of its deep-water amplitude over this depth (metres):
// none on the beach, all of it past uSwellDeep. 1 when off (uSwellByDepth 0,
// or no seabed: the deep sea).
uniform int   uSwellByDepth;
uniform float uSwellShallow;
uniform float uSwellDeep;

float swellDepthFactor(float depth) {
    return uSwellByDepth == 1 ? smoothstep(uSwellShallow, uSwellDeep, depth) : 1.0;
}
