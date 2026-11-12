// What the trees' fragment stages share (tree_mesh.frag, tree_impostor.frag):
// the crossover's dither, the colour as authored to linear light with the
// plant's tint, and the light on a leaf. Include once per shader with
// #pragma include.

#pragma include "../../atmosphere/shaders/display.glsl"

uniform vec3  uSunDir;     // toward the sun, render frame
uniform float uExposure;
uniform int   uTonemap;

// Interleaved gradient noise: a stable dither per pixel, in [0, 1).
float treeDither() {
    return fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
}

// The colour in linear light: the packs' textures hold a leaf's albedo as
// it is (their conifers' ~0.06: a real needle's), not sRGB (decoded, near
// black). A plant tinted 0.75 and up is dry and scorched (grey-brown);
// below, a little variety.
vec3 treeAlbedo(vec3 authored, float tint) {
    vec3 albedo = clamp(authored, 0.0, 1.0);
    if (tint >= 0.75) {
        float lum = dot(albedo, vec3(0.3, 0.5, 0.2));
        float burnt = (tint - 0.75) * 4.0;
        return mix(vec3(0.55, 0.42, 0.30), vec3(0.32, 0.30, 0.28), burnt) * lum * 1.6;
    }
    return albedo * mix(vec3(0.9, 0.95, 0.85), vec3(1.1, 1.05, 1.0), tint / 0.75);
}

// The light on a leaf (translucent: the sun wraps past the terminator), its
// sky light by how high in the tree it is (inner lower crowns see little),
// then the air to the camera, the exposure and the display encoding.
vec4 treeColor(vec3 albedo, vec3 N, vec3 up, vec3 sun, vec3 sky, vec3 airT, vec3 airS,
               float heightShare) {
    vec3 L = normalize(uSunDir);
    float wrap = clamp((dot(N, L) + 0.5) / 1.5, 0.0, 1.0);
    float ao = mix(0.65, 1.0, heightShare);
    vec3 C = albedo * (sun * wrap + sky * (0.7 + 0.3 * dot(N, up)) * ao);
    C = C * airT + airS;
    C *= uExposure;
    return vec4(uTonemap == 1 ? aces_tonemap(C) : pow(clamp(C, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
