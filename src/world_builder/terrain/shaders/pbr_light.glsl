// Physically based light on an opaque surface: the ground
// (TerrainSettings.pbr, terrain_frag.glsl) and a game's models (the PBR
// tiers of ModelSettings.shading, world_builder.objects). simplepbr's BRDF
// (panda3d-simplepbr 0.13, after Khronos' glTF-Sample-Viewer): GGX
// distribution, height-correlated Smith-GGX visibility, Schlick's Fresnel;
// glTF's metal and roughness. Include once per shader with #pragma include.
//
// In the world's units, not simplepbr's: objectLight (object_light.glsl)
// lights a white Lambert surface to the sunlight itself (no 1/pi), so the
// diffuse term here drops its 1/pi and the specular term takes pi: a rough
// dielectric is as bright as the terrain lights it.

const float PBR_PI = 3.141592653589793;
// Filament's floor: a mirror-like metal's sun glint stays a few pixels.
const float PBR_MIN_ROUGHNESS = 0.045;
const vec3  PBR_F0 = vec3(0.04);  // a dielectric's reflectance head-on

struct PbrSurface {
    vec3  diffuse;    // the albedo's diffuse share (none on a metal), linear
    vec3  f0;         // the reflectance head-on
    float roughness;  // perceptual, PBR_MIN_ROUGHNESS to 1
    float ao;         // ambient occlusion, on the sky light only
};

PbrSurface pbrSurface(vec3 albedo, float metallic, float roughness, float ao) {
    PbrSurface s;
    s.diffuse = albedo * (1.0 - metallic);
    s.f0 = mix(PBR_F0, albedo, metallic);
    s.roughness = clamp(roughness, PBR_MIN_ROUGHNESS, 1.0);
    s.ao = ao;
    return s;
}

// GGX / Trowbridge-Reitz, a = roughness squared.
float pbrDistribution(float NoH, float a) {
    float a2 = a * a;
    float f = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (PBR_PI * f * f);
}

// Smith-GGX, height-correlated: G / (4 N.L N.V).
float pbrVisibility(float NoL, float NoV, float a) {
    float a2 = a * a;
    float ggxv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float ggxl = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    float ggx = ggxv + ggxl;
    return ggx > 0.0 ? 0.5 / ggx : 0.0;
}

// Schlick, its power by a spherical gaussian (simplepbr's).
vec3 pbrFresnel(vec3 f0, float VoH) {
    return f0 + (vec3(1.0) - f0) * exp2((-5.55473 * VoH - 6.98316) * VoH);
}

// The specular reflectance under an even environment, integrated over the
// lobe (the split sum's second factor): Karis' analytic fit (Unreal's
// mobile EnvBRDFApprox), no lookup table.
vec3 pbrEnvironmentBrdf(vec3 f0, float roughness, float NoV) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
    vec2 ab = vec2(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

// The sunlight a surface sends to the camera: diffuse and specular, in
// objectLight's units (the sunlight arriving, N toward the camera's side).
vec3 pbrSun(PbrSurface s, vec3 N, vec3 L, vec3 V, vec3 sunlight) {
    float NoL = dot(N, L);
    if (NoL <= 0.0) {
        return vec3(0.0);
    }
    vec3 H = normalize(L + V);
    float NoV = clamp(abs(dot(N, V)), 1e-4, 1.0);
    float a = s.roughness * s.roughness;
    vec3 F = pbrFresnel(s.f0, clamp(dot(V, H), 0.0, 1.0));
    float DV = pbrDistribution(clamp(dot(N, H), 0.0, 1.0), a) * pbrVisibility(NoL, NoV, a);
    return (s.diffuse + F * (DV * PBR_PI)) * sunlight * NoL;
}

// How much of the environment's reflection an occluded surface keeps
// (Lagarde and de Rousiers, "Moving Frostbite to PBR"): 1 unoccluded.
float pbrSpecularOcclusion(float NoV, float ao, float roughness) {
    return clamp(pow(NoV + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao, 0.0, 1.0);
}

// The sky light a surface sends to the camera: diffuse from the sky as
// objectLight takes it (all of it facing up, a tenth facing down), and
// the environment env (the light along the reflected ray) reflected; the
// occlusion on both.
vec3 pbrAmbient(PbrSurface s, vec3 N, vec3 V, vec3 up, vec3 skylight, vec3 env) {
    float NoV = clamp(abs(dot(N, V)), 1e-4, 1.0);
    return s.diffuse * skylight * (0.55 + 0.45 * dot(N, up)) * s.ao
           + env * pbrEnvironmentBrdf(s.f0, s.roughness, NoV)
             * pbrSpecularOcclusion(NoV, s.ao, s.roughness);
}

// A tangent frame from the screen derivatives (Schüler, "Followup: normal
// mapping without precomputed tangents"), for a model without tangents.
// Out of any branch: it takes derivatives.
mat3 pbrCotangentFrame(vec3 N, vec3 p, vec2 uv) {
    vec3 dp1 = dFdx(p);
    vec3 dp2 = dFdy(p);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    vec3 dp2perp = cross(dp2, N);
    vec3 dp1perp = cross(N, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float scale = max(dot(T, T), dot(B, B));
    scale = scale > 0.0 ? inversesqrt(scale) : 0.0;
    return mat3(T * scale, B * scale, N);
}

// A tangent-space normal map's texel to a unit vector; a two-channel map
// (no blue: BC5) has its z made up.
vec3 pbrNormalTexel(vec3 texel) {
    vec3 n = texel * 2.0 - 1.0;
    if (texel.b < 0.01) {
        n.z = sqrt(clamp(1.0 - dot(n.xy, n.xy), 0.0, 1.0));
    }
    return n;
}
