// The forest on the ground's own pixels (terrain_frag.glsl): the far forest
// as Bruneton and Neyret's shader-map [BN12 §6, §7] (VEGETATION_PLAN.md,
// part V2), past the trees drawn one by one (world_builder.terrain.trees)
// and under them. Include once per shader with #pragma include, after
// terrain_light.glsl.
//
// A pixel of forest is split into sunlit ground, shadowed ground, sunlit
// crowns and shadowed crowns [BN12 §3, Eq. 1], their shares read from
// tables measured on patches of our own trees
// (scripts/bake_forest_tables.py, world_builder.terrain.far_forest), per far
// group (forest.FAR_GROUPS) and tree density lambda (the share of the
// trees' cells holding one, the vertex's forest bytes):
//   kg       the visible, sunlit ground          G(θv, θl, φ, λ)    [Eq. 14]
//   kg + kḡ  the visible ground                  G(θv, θv, 0, λ)    [Eq. 15]
//   T        the visible crowns' sunlit share    T(θv, θl, φ, λ)    [Eq. 16]
//   E        the visible crowns' sky visibility  E(θv, λ)           [Eq. 21]
//   D        the ground's sky visibility         D(λ)               [Eq. 18]
// Groups mixed in a pixel are taken as independent: the ground seen is
// seen through each (the products), the crowns' terms are the groups'
// averaged by their crowns' shares of the view. The shares are looked up
// per vertex (terrain.vert, forest_table.glsl: they vary slowly over the
// ground; per pixel they cost ~0.5 ms), the radiance summed per pixel:
//   Īg = the ground's sunlit light (objectLight's), J̄g = its sky light
//   times D [Eqs. 17, 19]; Īt = ρ F Lsun [Eq. 20], F the leaves' hotspot
//   1 - a1 (1 - exp(-a2 acos(v·l))) [Eq. 8]; J̄t = ρ E Lsky [Eq. 21].
// The transition [§7]: past the trees' range R the pixel is
//   kg Īg + (kg + kḡ) J̄g + (1 - kg - kḡ)(T Īt + J̄t)            [Eq. 13]
// and within it, as the impostors fade out over 0.8-1 R (their dither),
//   (1 - μ)(kg / (kg + kḡ) Īg + J̄g) + μ (T Īt + J̄t),
//   μ = (1 - kg - kḡ) smoothstep(0.8 R, R, s)                     [Eqs. 26-27]
// so the ground between the trees takes the visible ground's statistical
// shadow and sky occlusion (from the meshes' range out, where the trees
// cast none of their own), darker near the trees by the contrast term
// Δ = 1 - a3 (Γ - Λ A) [Eq. 10]. The density is the vertex's actual cover
// over the expected one times the groups' densities (Γ / A for Λ [§7]).
// Angles are taken from the local vertical, not the ground's normal (the
// trees stand vertical; Bruneton's slope transform is left out).

#pragma include "forest_table.glsl"

// The crowns' sunlit radiance per albedo and sunlight (their sunward sides'
// mean cosine, wrapped: leaves pass some light).
const float FOREST_CROWN_SUN = 0.75;

uniform float uTreeRange;        // the trees' impostors end here (0: none)
uniform float uMeshRange;        // and their meshes here, metres

// The contrast term's strength [Eq. 10].
const float FOREST_A3 = 0.6;

// The light a pixel of ground with forest sends to the camera.
//   forest      the vertex's forest bytes (cover / 2, the far groups' λ)
//   k, albedoD, hotCover   the forest's shares there (forestShares, per
//               vertex: terrain.vert)
//   dist        from the camera, metres
//   groundSun   the ground's own sunlit light (objectLight, no sky)
//   groundSky   its sky light (objectLight, no sun)
//   sunlight    the sunlight arriving (shadows of the terrain and clouds in),
//               times the sun's share past the terminator
//   skylight    the sky light arriving
//   variation   about 1: the crowns' colour from place to place
vec3 forestLight(vec4 forest, vec4 k, vec4 albedoD, vec2 hotCover, float dist,
                 vec3 groundSun, vec3 groundSky, vec3 sunlight, vec3 skylight,
                 float variation) {
    if (k.y >= 0.9999) {
        return groundSun + groundSky;  // no trees
    }
    float kg = k.x, kgs = k.y, T = k.z;
    // The crowns' radiances [Eqs. 8, 20, 21]; (1 + n·uz) / 2 on flat ground
    // is objectLight's 0.55 + 0.45 n·up, 1.
    vec3 rho = albedoD.rgb * variation;
    vec3 It = rho * sunlight * (hotCover.x * FOREST_CROWN_SUN);
    vec3 Jt = rho * skylight * k.w;
    // The ground's [Eqs. 10, 17, 19]: near, darker close to the trees.
    float R = uTreeRange;
    float nu = R > 0.0 ? smoothstep(0.8 * R, R, dist) : 1.0;
    float contrast = clamp(1.0 - FOREST_A3 * (forest.x * 2.0 - hotCover.y), 0.4, 1.4);
    vec3 Jg = groundSky * (albedoD.a * mix(contrast, 1.0, nu));
    // The visible ground's sunlit share: statistical from the meshes' range
    // out (nearer, the trees' own shadows: VEGETATION_PLAN.md V10).
    float sunShare = kgs > 1e-4 ? kg / kgs : 1.0;
    float statistical = R > 0.0 ? smoothstep(0.5 * uMeshRange, uMeshRange, dist) : 1.0;
    vec3 Ig = groundSun * mix(1.0, sunShare, statistical);
    float mu = (1.0 - kgs) * nu;
    return (1.0 - mu) * (Ig + Jg) + mu * (T * It + Jt);
}
