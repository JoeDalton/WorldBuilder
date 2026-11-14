// A game's model lit and displayed as the world is (world_builder.objects,
// docs/source/object_lighting.md): its colour in linear light, lit by the
// sunlight at the pixel in every world shadow (model_light.glsl) and the
// sky light at the model (its own Panda3D ambient light, World.track_light),
// then the air in front, the exposure and the tonemap.
//
// The body of every shading tier (ModelSettings.shading): an entry file
// (model_<tier>.frag) defines the tier's features and includes it.
//   (none)                as the terrain lights its ground (objectLight)
//   MODEL_PBR             metal and roughness (model_pbr.glsl), emission
//   MODEL_NORMAL_MAP      the models' normal maps
//   MODEL_SKY_REFLECTION  the sky reflected from the sky-view table

#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "../../atmosphere/shaders/display.glsl"
#pragma include "../../clouds/shaders/cloud_field.glsl"
#pragma include "../../terrain/shaders/object_light.glsl"
#pragma include "model_light.glsl"
#ifdef MODEL_PBR
#pragma include "model_pbr.glsl"
#endif
#ifdef MODEL_SKY_REFLECTION
#pragma include "../../atmosphere/shaders/sky_view.glsl"
#endif

// The base colour's texture stage (white without one).
uniform sampler2D p3d_TextureModulate;
uniform vec4 p3d_ColorScale;
// Without a material: baseColor 0 and diffuse 1, roughness 1 (matte),
// metallic 0, emission 0. A PBR material's diffuse is its baseColor times
// (1 - metallic), an older one's baseColor its diffuse: the larger of the
// two is the base colour in every case.
uniform struct {
    vec4 baseColor;
    vec4 diffuse;
    vec4 emission;
    float roughness;
    float metallic;
} p3d_Material;
// The sum of the node's ambient lights: the sky light where it is.
uniform struct {
    vec4 ambient;
} p3d_LightModel;
uniform vec3  uSunDir;      // toward the sun, render frame
uniform float uExposure;
uniform float uLightScale;  // World.light_scale, taken back off the ambient
// 1: the model is authored in linear light (glTF: its colour textures are
// sRGB, decoded by the GPU; its factors linear); 0: as displayed (decoded
// here). Set per part by world_builder.objects.mark_linear.
uniform int   uBaseLinear;

#ifdef MODEL_PBR
// glTF's metal-roughness texture: G roughness, B metal (white without).
uniform sampler2D p3d_TextureSelector;
uniform sampler2D p3d_TextureEmission;   // white without
#endif
#ifdef MODEL_NORMAL_MAP
uniform sampler2D p3d_TextureNormal;     // flat without
in vec4 vTangent;                         // render frame, w its handedness
#endif

in vec3 vRelPos;
in vec3 vNormal;
in vec2 vUv;
in vec4 vColor;

out vec4 fragColor;

#ifdef MODEL_SKY_REFLECTION
// The sky along the reflected ray R from the sky-view table, inside an
// air (else the even sky, hemi): sharp on a smooth surface, the even sky
// on a rough one (the table has no blurred levels) and below the horizon
// (the table holds the air there, not the ground).
vec3 skyReflection(vec3 R, vec3 up, vec3 L, float roughness, vec3 hemi) {
    vec3 transmittance, inscatter;
    if (!skyViewAlong(R, L, transmittance, inscatter)) {
        return hemi;
    }
    float sharp = (1.0 - smoothstep(0.15, 0.75, roughness))
                * smoothstep(-0.02, 0.05, dot(R, up));
    return mix(hemi, inscatter, sharp);
}
#endif

void main() {
    vec4 texel = texture(p3d_TextureModulate, vUv);
    vec4 factor = vColor * p3d_ColorScale * max(p3d_Material.baseColor, p3d_Material.diffuse);
    vec4 base = texel * factor;
    vec3 albedo = uBaseLinear != 0 ? clamp(base.rgb, 0.0, 1.0)
                                   : pow(clamp(base.rgb, 0.0, 1.0), vec3(2.2));
    float dist = max(length(vRelPos), 1e-3);
    vec3 V = -vRelPos / dist;
    vec3 Ng = length(vNormal) > 1e-12 ? normalize(vNormal) : V;  // the geometry's
    vec3 N = Ng;
#ifdef MODEL_NORMAL_MAP
    // Out of any branch (derivatives): the frame from the screen, for a
    // model without tangents (they read 0).
    mat3 tbn = pbrCotangentFrame(Ng, vRelPos, vUv);
    vec3 T = vTangent.xyz - Ng * dot(Ng, vTangent.xyz);
    if (dot(T, T) > 1e-12) {
        T = normalize(T);
        tbn = mat3(T, cross(Ng, T) * (vTangent.w < 0.0 ? -1.0 : 1.0), Ng);
    }
    N = normalize(tbn * pbrNormalTexel(texture(p3d_TextureNormal, vUv).rgb));
#endif
    if (!gl_FrontFacing) {
        N = -N;  // a two-sided model's back
        Ng = -Ng;
    }
    vec3 L = normalize(uSunDir);
    // Taken out of any branch.
    float footprint = max(length(dFdx(vRelPos)), length(dFdy(vRelPos)));
    // The shadows only where the sun can light the pixel at all (the
    // terms go as N.L): the world's, and the objects' (its own parts' too).
    vec3 sunlight = vec3(0.0);
    if (dot(N, L) > 0.0) {
        sunlight = modelSunlight(vRelPos, L, footprint) * casterShadow(vRelPos, Ng, L);
    }
    vec3 skylight = p3d_LightModel.ambient.rgb / uLightScale;
    vec3 up = modelUp(vRelPos, N);
#ifdef MODEL_PBR
    vec4 metalRough = texture(p3d_TextureSelector, vUv);
    PbrSurface s = pbrSurface(albedo, clamp(p3d_Material.metallic * metalRough.b, 0.0, 1.0),
                              p3d_Material.roughness * metalRough.g, 1.0);
    vec3 R = reflect(-V, N);
    // The environment: the sky light, as on a surface facing the ray.
    vec3 env = skylight * (0.55 + 0.45 * dot(R, up));
#ifdef MODEL_SKY_REFLECTION
    env = skyReflection(R, up, L, s.roughness, env);
#endif
    vec3 C = pbrSun(s, N, L, V, sunlight) + pbrAmbient(s, N, V, up, skylight, env);
    // A glowing part: its emission as linear light, before the air.
    C += p3d_Material.emission.rgb * texture(p3d_TextureEmission, vUv).rgb;
#else
    vec3 C = objectLight(albedo, N, p3d_Material.roughness, 1.0, up, L, V, sunlight, 1.0,
                         skylight);
#endif
    // The air between the camera and the model, as over the ground.
    C = applyAtmospheres(C, vRelPos / dist, dist, L);
    fragColor = vec4(aces_tonemap(C * uExposure), base.a);
}
