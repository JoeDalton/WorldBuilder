#version 330
// A game's model lit and displayed as the world is (world_builder.objects,
// docs/source/object_lighting.md): its colour decoded to linear light, lit
// as the terrain lights its ground (object_light.glsl) by the sunlight at
// the pixel in every world shadow (model_light.glsl) and the sky light at
// the model (its own Panda3D ambient light, World.track_light), then the
// air in front, the exposure and the tonemap.

#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "../../atmosphere/shaders/display.glsl"
#pragma include "../../clouds/shaders/cloud_field.glsl"
#pragma include "../../terrain/shaders/object_light.glsl"
#pragma include "model_light.glsl"

uniform sampler2D p3d_Texture0;        // white without a texture
uniform vec4 p3d_ColorScale;
// Without a material: diffuse 1, roughness 1 (matte).
uniform struct {
    vec4 diffuse;
    float roughness;
} p3d_Material;
// The sum of the node's ambient lights: the sky light where it is.
uniform struct {
    vec4 ambient;
} p3d_LightModel;
uniform vec3  uSunDir;      // toward the sun, render frame
uniform float uExposure;
uniform float uLightScale;  // World.light_scale, taken back off the ambient

in vec3 vRelPos;
in vec3 vNormal;
in vec2 vUv;
in vec4 vColor;

out vec4 fragColor;

void main() {
    // Colours are authored as displayed: decoded to linear light.
    vec4 base = texture(p3d_Texture0, vUv) * vColor * p3d_ColorScale * p3d_Material.diffuse;
    vec3 albedo = pow(clamp(base.rgb, 0.0, 1.0), vec3(2.2));
    float dist = max(length(vRelPos), 1e-3);
    vec3 V = -vRelPos / dist;
    vec3 N = length(vNormal) > 1e-12 ? normalize(vNormal) : V;
    if (!gl_FrontFacing) {
        N = -N;  // a two-sided model's back
    }
    vec3 L = normalize(uSunDir);
    // Taken out of any branch.
    float footprint = max(length(dFdx(vRelPos)), length(dFdy(vRelPos)));
    // The shadows only where the sun can light the pixel at all
    // (objectLight's terms go as N.L): the world's, and the objects' (its
    // own parts' too).
    vec3 sunlight = vec3(0.0);
    if (dot(N, L) > 0.0) {
        sunlight = modelSunlight(vRelPos, L, footprint) * casterShadow(vRelPos, N, L);
    }
    vec3 skylight = p3d_LightModel.ambient.rgb / uLightScale;
    vec3 C = objectLight(albedo, N, p3d_Material.roughness, 1.0, modelUp(vRelPos, N),
                         L, V, sunlight, 1.0, skylight);
    // The air between the camera and the model, as over the ground.
    C = applyAtmospheres(C, vRelPos / dist, dist, L);
    fragColor = vec4(aces_tonemap(C * uExposure), base.a);
}
