#version 330
// The cloud shadow map's bake (clouds.shadow_map): per texel, the clouds'
// optical depth toward the sun along the ray that enters the slab's base
// there (cloud_field.glsl's cloudShadowDepth from that point), the march
// once per texel instead of once per pixel of ground. Each texel is a node
// of an azimuthal equidistant chart on the base's sphere (uMapChart: its
// centre's up and the base's radius, east and the spacing, north and the
// nodes per side).
#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "cloud_field.glsl"

uniform int  uCloudShadowLayer;
uniform int  uMapSteps;
uniform vec4 uMapChart[4];

out vec4 fragColor;

void main() {
    float size = uMapChart[2].w;
    vec2 xy = (gl_FragCoord.xy - 0.5 * size) * uMapChart[1].w;
    float distance = length(xy);
    float angle = distance / uMapChart[0].w;
    vec2 toward = distance > 0.0 ? xy / distance : vec2(0.0);
    vec3 dir = cos(angle) * uMapChart[0].xyz
             + sin(angle) * (toward.x * uMapChart[1].xyz + toward.y * uMapChart[2].xyz);
    Field f = getField(uCloudShadowLayer);
    // On the base's sphere: the march starts there (enter 0).
    vec3 start = dir * (planetRadius + f.slabBase);
    fragColor = vec4(cloudShadowDepth(f, start, sunDir, uMapSteps), 0.0, 0.0, 1.0);
}
