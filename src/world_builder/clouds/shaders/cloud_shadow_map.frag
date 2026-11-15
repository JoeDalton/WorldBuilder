#version 330
// The cloud shadow map's bake (clouds.shadow_map): per texel, the genera's
// summed optical depth toward the sun along the ray that enters the sky
// bands' base there (cloud_field.glsl's skyShadowLevels from that point: each
// band's chord cut at its clouds' own top, every genus in it marched by its
// own field and weather), the march once per texel instead of once per pixel
// of ground. A texel holds the total (r) and the depth a quarter, a half and
// three quarters of the way along the bands' summed chord (g, b, a), for the
// clouds' own lighting (cloudMapLevelsAt); the ground reads r. Each texel is
// a node of an azimuthal equidistant chart on the base's sphere (uMapChart:
// its centre's up and the base's radius, east and the spacing, north and the
// nodes per side).
#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "cloud_field.glsl"

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
    // On the base's sphere: the march starts there (enter 0).
    vec3 start = dir * uMapChart[0].w;
    vec3 levels;
    float total = skyShadowLevels(start, sunDir, uMapSteps, 1.0, true, true, levels);
    fragColor = vec4(total, levels);
}
