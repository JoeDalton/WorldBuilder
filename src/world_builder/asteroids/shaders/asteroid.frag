#version 330

// A rock's surface: its prototype's normal map (the full-detail shape) turned
// with it, the terrain's rock and snow textures blended by its ice
// (terrain_material.glsl: rock projected from the sides on steep flanks),
// lit as the terrain lights its ground (object_light.glsl), then either the
// world's exposure and tonemap or a plain display encoding (for scenes lit
// by Panda's fixed-function lights). Rocks fading out at their drawing
// distance are dithered away.

#pragma include "../../atmosphere/shaders/display.glsl"
#pragma include "../../terrain/shaders/terrain_material.glsl"
#pragma include "../../terrain/shaders/object_light.glsl"

uniform sampler2DArray uRockNormals;  // six layers per prototype
uniform mat3 uFieldRot;
uniform vec3 uMeanAlbedo[6];          // linear; drawn instead of textures far away
uniform float uMeanHeight[6];         // the textures' mean height
uniform float uDetailFar;             // textures fade to the mean colour by here
uniform vec3 uSunDir;                 // toward the sun, render frame
uniform vec3 uSunColor;
uniform vec3 uAmbient;
uniform float uExposure;
uniform int uTonemap;                 // 1: aces_tonemap, 0: plain encoding
uniform float uRockUvPeriod;          // rock.ROCK_UV_PERIOD

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
    vec3 body = texture(uRockNormals, layer).rgb * 2.0 - 1.0;
    vec3 N = normalize(uFieldRot * qrot(vQuat, body));
    vec3 up = normalize(vUp);

    // Rock and snow, the two first layers (biomes.BIOMES).
    float w[6] = float[6](ice, 1.0 - ice, 0.0, 0.0, 0.0, 0.0);
    vec3 meanAlbedo = meanLayeredAlbedo(w, uMeanAlbedo, uMeanHeight);
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
    fragColor = vec4(uTonemap == 1 ? aces_tonemap(C)
                                   : pow(clamp(C, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
