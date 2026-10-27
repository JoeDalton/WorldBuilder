#version 330

// A rock's surface: its prototype's normal map (the full-detail shape) turned
// with it, the terrain's rock and snow textures where its ice patches lie
// (the normal map's fourth channel, shape.IcePatches: ice on the share of
// the surface ranked above 1 - its ice; terrain_material.glsl: rock
// projected from the sides on steep flanks),
// lit as the terrain lights its ground (object_light.glsl), then either the
// world's exposure and tonemap or a plain display encoding (for scenes lit
// by Panda's fixed-function lights). The field's haze between the camera
// and the rock veils it (haze.glsl): far rocks melt into the haze they leave
// to. Rocks fading out at their drawing distance are dithered away.

#pragma include "../../atmosphere/shaders/display.glsl"
#pragma include "../../terrain/shaders/terrain_material.glsl"
#pragma include "../../terrain/shaders/object_light.glsl"

uniform sampler2DArray uRockNormals;  // six layers per prototype
uniform mat3 uFieldRot;
uniform vec3 uMeanAlbedo[6];          // linear; drawn instead of textures far away
uniform float uNormalSize;            // the normal maps' texels per face side
uniform float uDetailFar;             // textures fade to the mean colour by here
uniform vec3 uSunDir;                 // toward the sun, render frame
uniform vec3 uSunColor;
uniform vec3 uAmbient;
uniform float uExposure;
uniform int uTonemap;                 // 1: aces_tonemap, 0: plain encoding
uniform float uRockUvPeriod;          // rock.ROCK_UV_PERIOD
uniform int uVeil;                    // 1: veiled by the field's haze

#pragma include "haze.glsl"

in vec3 vRelPos;
in vec3 vNormal;
in vec3 vUp;
in vec3 vFaceUv;
in float vHeight;
flat in vec3 vTu;
flat in vec3 vTv;
flat in vec4 vQuat;
flat in vec4 vRock;

out vec4 fragColor;

const float PI = 3.141592653589793;
const int VEIL_SAMPLES = 8;
const float PATCH_EDGE = 0.02;  // shape.PATCH_EDGE

// shape.ice_weight: ice above rank 1 - ice, a narrow ramp between.
float iceWeight(float rank, float ice) {
    if (ice <= 0.0) return 0.0;
    if (ice >= 1.0) return 1.0;
    return smoothstep(1.0 - ice - PATCH_EDGE, 1.0 - ice + PATCH_EDGE, rank);
}

vec3 qrot(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.yzw, v);
    return v + q.x * t + cross(q.yzw, t);
}

// 4x4 ordered dither, in (0, 1).
float bayer(vec2 pixel) {
    const float M[16] = float[16](0., 8., 2., 10., 12., 4., 14., 6.,
                                  3., 11., 1., 9., 15., 7., 13., 5.);
    ivec2 p = ivec2(mod(pixel, 4.0));
    return (M[p.y * 4 + p.x] + 0.5) / 16.0;
}

void main() {
    if (vRock.w < 1.0 && vRock.w <= bayer(gl_FragCoord.xy)) discard;
    float radius = vRock.x;
    float ice = vRock.y;
    float dist = length(vRelPos);
    int face = int(vFaceUv.z + 0.5);

    // The full-detail normal, from the body frame.
    vec3 layer = vec3((vFaceUv.xy + 1.0) * 0.5, float(int(vRock.z + 0.5) * 6 + face));
    vec4 normalRank = texture(uRockNormals, layer);
    vec3 body = normalRank.rgb * 2.0 - 1.0;
    vec3 N = normalize(uFieldRot * qrot(vQuat, body));
    vec3 up = normalize(vUp);

    // Snow on the rock's ice patches, rock elsewhere (biomes.BIOMES' two
    // first layers). Where a texel covers many pixels' worth of patches
    // (far: the mipmaps average the ranks), the ice's share instead.
    vec2 texels = (vFaceUv.xy + 1.0) * (0.5 * uNormalSize);
    float footprint = max(length(dFdx(texels)), length(dFdy(texels)));
    float snow = mix(iceWeight(normalRank.a, ice), ice, smoothstep(4.0, 16.0, footprint));
    float w[6] = float[6](snow, 1.0 - snow, 0.0, 0.0, 0.0, 0.0);
    vec3 meanAlbedo = mix(uMeanAlbedo[1], uMeanAlbedo[0], snow);
    float far = smoothstep(0.6 * uDetailFar, uDetailFar, dist);
    vec3 albedo = meanAlbedo;
    vec3 surfaceN = N;
    float roughness = 0.85;
    float ao = 1.0;
    if (far < 1.0) {
        // Texture metres along the face, as the rock's terrain lays them
        // (cubesphere.grid_for): a whole number of periods across a face
        float span = max(1.0, floor(0.5 * PI * radius / uRockUvPeriod + 0.5)) * uRockUvPeriod;
        vec2 uv = (vFaceUv.xy + 1.0) * (0.5 * span);
        vec3 Tu = normalize(vTu - up * dot(up, vTu));
        vec3 Tv = normalize(vTv - up * dot(up, vTv));
        layeredMaterial(w, uv, vHeight, N, Tu, Tv, dot(N, up), dist, meanAlbedo, far,
                        albedo, surfaceN, roughness, ao);
    }

    vec3 L = normalize(uSunDir);
    vec3 V = normalize(-vRelPos);
    // A rock has no "up" for the sky light: it lights every face alike.
    vec3 C = objectLight(albedo, surfaceN, roughness, ao, surfaceN, L, V,
                         uSunColor, 1.0, uAmbient);
    C *= uExposure;
    vec3 rgb = uTonemap == 1 ? aces_tonemap(C) : pow(clamp(C, 0.0, 1.0), vec3(1.0 / 2.2));
    if (uVeil == 1) {
        // The haze from the camera to here, over the rock as over the sky.
        vec3 dirRender = normalize(vRelPos);
        vec3 d = transpose(uFieldRot) * dirRender;
        vec2 s0, s1;
        int n = hazeIntervals(uCamField, d, s0, s1);
        float tau = 0.0;
        if (n > 0) tau += hazeDepth(uCamField, d, s0.x, min(s0.y, dist), VEIL_SAMPLES);
        if (n > 1) tau += hazeDepth(uCamField, d, s1.x, min(s1.y, dist), VEIL_SAMPLES);
        rgb = mix(hazeColour(dirRender), rgb, exp(-tau));
    }
    fragColor = vec4(rgb, 1.0);
}
