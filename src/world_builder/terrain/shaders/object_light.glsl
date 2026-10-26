// Sunlight and sky light on an opaque surface, as the terrain lights its
// ground: shared by the terrain (terrain_frag.glsl) and the objects drawn
// in its light. Linear radiance in, before the exposure and the tonemap
// (display.glsl's aces_tonemap). Include once per shader with
// #pragma include.

// The light a surface sends to the camera: Lambert and a gloss highlight
// from the roughness under the sun, and the sky light from above.
//   albedo     linear
//   N          the shading normal (normal-mapped)
//   roughness  0 mirror-like to 1 matte
//   ao         ambient occlusion, on the sky light only
//   up         the local vertical: the sky lights a surface facing it fully
//              and one facing away at 0.1
//   L, V       toward the sun and toward the camera
//   sunlight   the sunlight arriving there
//   sunDay     the sun's share past any terminator (1 = fully up)
//   skylight   the sky light arriving there
vec3 objectLight(vec3 albedo, vec3 N, float roughness, float ao, vec3 up, vec3 L, vec3 V,
                 vec3 sunlight, float sunDay, vec3 skylight) {
    vec3 H = normalize(L + V);
    float lambert = max(dot(N, L), 0.0) * sunDay;
    float gloss = 1.0 - roughness;
    float specular = gloss * gloss * 0.3 * pow(max(dot(N, H), 0.0), mix(8.0, 96.0, gloss));
    return albedo * (sunlight * lambert + skylight * (0.55 + 0.45 * dot(N, up)) * ao)
           + sunlight * specular * lambert;
}
