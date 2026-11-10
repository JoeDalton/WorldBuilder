// The vegetation on the ground's own pixels (terrain_frag.glsl), where the
// canopy's shell (canopy_frag.glsl) draws no plant, and the plants' shadows
// on the ground near it. Include once per shader with #pragma include, after
// terrain_light.glsl.
//
// Two looks:
//
// - Near (VEG_NEAR: the variant of the ground's shader on the chunks with a
//   shell, terrain_veg.frag): the plants' shadows on the ground, within
//   uVegShadowRange. Each ground pixel looks toward the sun for the one
//   crown between it and the sun (the cell its sun ray crosses at a crown's
//   height: vegetation_cells.glsl's placement, the plants the shell draws),
//   its ellipse cast on the ground. Only there: the placement's code in the
//   ground's shader cost every ground pixel (its occupancy) ~0.5 ms.
// - Far (past the shell): the canopy as a statistical cover over the ground
//   (vegCoverage: the crowns' share seen from above), seen through the gaps
//   between crowns. Along a direction at zenith cosine mu a crown blocks
//   more than its plan area (its height too): the gaps' share is
//   exp(-sum cover * G(mu) / mu), with G the crowns' projected area per plan
//   area (an ellipsoid's: sqrt(mu^2 + (b / r)^2 (1 - mu^2))). The same along
//   the sun gives the ground's sunlit share; toward the sun's own direction
//   (the hot spot) the gaps seen are the lit ones.

#pragma include "vegetation_cells.glsl"

uniform int   uVegOn;            // 0: no vegetation (a barren planet, off)
uniform float uVegRange;         // the shell draws plants out to here, metres
                                 // (0: none, the far look everywhere)
uniform float uVegShadowRange;   // the plants' shadows on the ground reach
                                 // this far, metres (0: none)

// (b / r)^2 of a tree's crown and of a bush's at mean sizes (plantCrown's),
// a tree's crown seen from the side standing on its trunk, a little more.
const float VEG_TREE_ASPECT2 = 3.2;
const float VEG_BUSH_ASPECT2 = 0.7;
// The crown height the impostors and the shadows look at, metres: a mean
// tree's crown centre (plantCrown's of a mean tree), and a bush's.
const float VEG_TREE_MID = 10.2;
const float VEG_BUSH_MID = 0.6;

// The gaps' share along a direction of zenith cosine mu, under a cover of
// trees and bushes (vegCoverage's).
float vegGaps(vec2 cover, float mu) {
    mu = max(mu, 0.03);
    float bushes = max(cover.x - cover.y, 0.0);
    float mu2 = mu * mu;
    float depth = (cover.y * sqrt(mu2 + VEG_TREE_ASPECT2 * (1.0 - mu2))
                   + bushes * sqrt(mu2 + VEG_BUSH_ASPECT2 * (1.0 - mu2))) / mu;
    return exp(-depth);
}

#ifdef VEG_NEAR
// The plant whose crown a ray from ground point uv (uv metres) toward dir
// (its horizontal part in uv per metre of height: dirUv) meets at height
// mid: its cell's plant, and where on its crown's disc the ray passes
// (in crown radii; |at| > 1: it misses). The plant's own crown height
// replaces mid for the hit test.
bool crownAlong(vec2 uv, vec2 dirUv, float mid, ivec2 period, int face, vec4 w,
                out Plant p, out vec2 at) {
    vec2 probe = uv + dirUv * mid;
    p = plantAt(ivec2(floor(probe / VEG_CELL)), period, face, w);
    if (p.kind == VEG_NONE) {
        return false;
    }
    vec2 crown = plantCrown(p);
    at = (uv + dirUv * crown.x - p.uv) / p.radius;
    return dot(at, at) < 1.0;
}
#endif

// The vegetation's share of a ground pixel's look.
//   w         the biome weights (terrain_frag.glsl's, wobbled)
//   uv        the pixel's texture coordinate, metres; period, face: its
//             chunk's (vegetation_cells.glsl's plantAt)
//   Tu, Tv    the face's texture axes on the tangent plane (render frame)
//   up, L, V  the vertical, toward the sun, toward the camera
//   dist      from the camera, metres; footprint: the ground a pixel spans
//   variation about 1: the cover's colour from place to place (the ground's
//             noise)
// Changes the albedo (the cover over the ground), the sunlight (the crowns'
// shadows) and the sky light's occlusion; the normal and the roughness are
// left (kept for a look that changes them).
void vegetationGround(float w[6], vec2 uv, vec2 period, int face, vec3 Tu, vec3 Tv,
                      vec3 up, vec3 L, vec3 V, float dist, float footprint,
                      float variation,
                      inout vec3 albedo, inout vec3 normal, inout vec3 sunlight,
                      inout float roughness, inout float ao) {
    if (uVegOn == 0) {
        return;
    }
    vec4 weights = vec4(w[3], w[2], w[4], w[1]);  // forest, grass, desert, rock
    vec2 cover = vegCoverage(weights);
    if (cover.x <= 1e-4) {
        return;
    }
    ivec2 per = ivec2(period + 0.5);
    float muV = dot(V, up);
    float muL = dot(L, up);
    // Where the shell draws the plants, the ground is the ground; past it,
    // the impostors, then the cover, take over (over the shell's fade).
    float beyond = uVegRange > 0.0 ? smoothstep(0.7 * uVegRange, uVegRange, dist) : 1.0;
    float mid = cover.y > 0.25 * cover.x ? VEG_TREE_MID : VEG_BUSH_MID;

    float nearAo = 1.0 - 0.55 * min(cover.x, 1.0);
    if (beyond <= 0.0) {
        // Near: the ground between the drawn plants, shaded by their cover
        // (the sky they hide) and their shadows.
#ifdef VEG_NEAR
        if (muL > 0.02 && dist < uVegShadowRange) {
            Plant p;
            vec2 at;
            vec2 sunUv = vec2(dot(L, Tu), dot(L, Tv)) / max(muL, 0.05);
            if (crownAlong(uv, sunUv, mid, per, face, weights, p, at)) {
                float fade = smoothstep(0.7 * uVegShadowRange, uVegShadowRange, dist);
                sunlight *= mix(smoothstep(0.75, 1.0, length(at)), 1.0, fade);
            }
        }
#endif
        ao *= nearAo;
        return;
    }

    // The cover.
    float seen = 1.0 - vegGaps(cover, muV);     // the canopy's share of the pixel
    float sunGaps = muL > 0.0 ? vegGaps(cover, muL) : 0.0;
    float hot = pow(max(dot(V, L), 0.0), 120.0);
    float groundLit = mix(sunGaps, 1.0, hot);
    // The crowns' own lit share as seen: their sunward sides face the camera
    // when it looks down-sun, and a dense canopy shades its lower crowns.
    float crownLit = (0.95 + 0.3 * dot(V, L)) * mix(0.8, 1.0, sunGaps);
    vec3 farAlbedo = mix(albedo, canopyMeanColor(weights) * variation, seen);
    float farSun = mix(groundLit, crownLit, seen);
    float farAo = mix(nearAo, 0.95, seen);

    albedo = mix(albedo, farAlbedo, beyond);
    sunlight *= mix(1.0, farSun, beyond);
    ao *= mix(nearAo, farAo, beyond);
}
