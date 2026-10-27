// The terrain's textured materials: the six biome layers (biomes.BIOMES) of
// world_builder.terrain.materials, blended by weight and height, with rock
// also projected from the sides on steep ground. Shared by the terrain
// (terrain_frag.glsl) and anything drawn with its textures. Include once per
// shader with #pragma include.

uniform sampler2DArray uAlbedoH;  // sRGB albedo + height
uniform sampler2DArray uNormRA;   // normal xy, roughness, ao
uniform float uTileM[6];          // metres per texture repeat
uniform float uBlendDepth;        // biome transition softness: ~0.1 crisp
                                  // height-cut edges, >= 1 smooth crossfades

const int ROCK = 1;
const float MACRO_SCALE = 8.0;  // must match materials.MACRO_SCALE

// Samples inside the per-layer branches below are in non-uniform control flow,
// where implicit derivatives are undefined: pass them explicitly.
vec4 sampleLayer(sampler2DArray s, vec2 uv, vec2 gx, vec2 gy, float scale, int layer) {
    return textureGrad(s, vec3(uv * scale, float(layer)), gx * scale, gy * scale);
}

// The textured surface where the biome weights are w (summing to 1): uv are
// the texture coordinates in metres, height the ground's (the side
// projections' second axis), N the geometric normal, Tu and Tv the face's
// texture axes on the tangent plane, upN the cosine of the slope, dist the
// distance from the camera; meanAlbedo is what the textures fade to by far
// (1 = wholly). Returns the albedo (linear), the normal-mapped normal, the
// roughness and the ambient occlusion.
void layeredMaterial(float w[6], vec2 uv, float height, vec3 N, vec3 Tu, vec3 Tv,
                     float upN, float dist, vec3 meanAlbedo, float far,
                     out vec3 albedo, out vec3 surfaceN, out float roughness,
                     out float ao) {
    // More of the large repeat with distance: hides tiling, then takes over.
    float macro = mix(0.3, 0.75, smoothstep(20.0, 600.0, dist));
    float normalFade = 1.0 - smoothstep(60.0, 600.0, dist);
    // Steep ground: rock is projected from the sides too, not stretched.
    vec3 tri = pow(abs(vec3(dot(N, Tu), dot(N, Tv), upN)), vec3(4.0));
    tri /= tri.x + tri.y + tri.z;

    vec2 gx = dFdx(uv), gy = dFdy(uv);
    vec2 sideX = vec2(uv.y, height), sideY = vec2(uv.x, height);
    vec2 sxGx = dFdx(sideX), sxGy = dFdy(sideX);
    vec2 syGx = dFdx(sideY), syGy = dFdy(sideY);

    vec4 layerAlbedoH[6];
    vec4 layerNRA[6];
    float best = -1.0;
    for (int i = 0; i < 6; ++i) {
        if (w[i] < 0.02) {
            layerAlbedoH[i] = vec4(0.0);
            layerNRA[i] = vec4(0.5, 0.5, 0.85, 1.0);
            continue;
        }
        float scale = 1.0 / uTileM[i];
        vec4 ah = mix(sampleLayer(uAlbedoH, uv, gx, gy, scale, i),
                      sampleLayer(uAlbedoH, uv, gx, gy, scale / MACRO_SCALE, i), macro);
        vec4 nra = sampleLayer(uNormRA, uv, gx, gy, scale, i);
        if (i == ROCK && tri.z < 0.98) {
            ah = ah * tri.z
               + sampleLayer(uAlbedoH, sideX, sxGx, sxGy, scale, i) * tri.x
               + sampleLayer(uAlbedoH, sideY, syGx, syGy, scale, i) * tri.y;
            nra = mix(vec4(0.5, 0.5, nra.b, nra.a), nra, tri.z);
        }
        layerAlbedoH[i] = ah;
        layerNRA[i] = nra;
        best = max(best, ah.a + w[i]);
    }
    // Height blend: where two biomes meet, the locally taller texels win
    // (rock pokes through snow, sand fills the hollows). A large depth
    // lets every layer within it contribute, in proportion to its weight:
    // a smooth crossfade with only a hint of the height structure.
    float sum = 0.0;
    vec3 color = vec3(0.0);
    vec4 nra = vec4(0.0);
    for (int i = 0; i < 6; ++i) {
        if (w[i] < 0.02) continue;
        float b = max(layerAlbedoH[i].a + w[i] - (best - uBlendDepth), 0.0) * w[i];
        color += b * layerAlbedoH[i].rgb;
        nra += b * layerNRA[i];
        sum += b;
    }
    color /= max(sum, 1e-5);
    nra /= max(sum, 1e-5);
    albedo = mix(color, meanAlbedo, far);
    roughness = nra.b;
    ao = mix(nra.a, 1.0, far);

    vec2 txy = (nra.xy * 2.0 - 1.0) * normalFade;
    vec3 tn = vec3(txy, sqrt(max(0.0, 1.0 - dot(txy, txy))));
    vec3 T = normalize(Tu - N * dot(N, Tu));
    vec3 B = cross(N, T);
    surfaceN = normalize(T * tn.x + B * tn.y + N * tn.z);
}
