#version 330
// The clouds' shadow on the scene, once a frame at the depth passes'
// resolution (clouds.march.GroundShadow): what the terrain's and the sea's
// pixels took per pixel (terrain_frag.glsl's cloudShade, planetCloudShadow),
// for the scene the march's depth passes drew. There each pixel's work ran
// for every pixel of the 2 x 2 quads of the ground's small triangles (0.8-0.9
// ms a frame at 2560 x 1440, performance.md §33); here it runs once per
// texel, and the ground reads it back by its distance (cloudShadeAt).
//
// Out: r the sun's share the clouds leave, g the scene's distance there in
// kilometres (a half float: a part in two thousand), -1 where the depth
// passes drew nothing (the sky).
#pragma include "../../atmosphere/shaders/atmosphere.glsl"
#pragma include "cloud_field.glsl"

in vec2 vNdc;
out vec4 fragColor;

uniform sampler2D uSceneDepthNear;
uniform sampler2D uSceneDepthFar;
uniform int   uNearDepthOn;
uniform mat4  uClipToRender;     // the near range's clip space -> the render frame
uniform mat4  uClipToRenderFar;  // the far range's
uniform vec3  uCamRender;        // camera, render frame
uniform vec3  camPos;            // camera, planet frame
uniform int   uOwnAtmosphere;    // the planet's atmosphere, -1 none: no shadow
uniform vec2  uViewSize;         // this pass's size, pixels (the depth passes')
uniform int   uDepthScale;       // view pixels per texel, per axis

const float KM = 1000.0;

vec3 unproject(mat4 clipToRender, vec3 ndc) {
    vec4 p = clipToRender * vec4(ndc, 1.0);
    return p.xyz / p.w;
}

void main() {
    ivec2 texel = ivec2(gl_FragCoord.xy);
    vec2 ndc = gl_FragCoord.xy / uViewSize * 2.0 - 1.0;
    float nearDepth = uNearDepthOn != 0 ? texelFetch(uSceneDepthNear, texel, 0).r : 1.0;
    float farDepth = texelFetch(uSceneDepthFar, texel, 0).r;
    bool near = nearDepth < 1.0;
    mat4 toRender = near ? uClipToRender : uClipToRenderFar;
    vec3 end = unproject(toRender, vec3(ndc, (near ? nearDepth : farDepth) * 2.0 - 1.0));
    vec3 rel = end - uCamRender;
    // Taken out of any branch (derivatives): this pass's own footprint.
    float local = max(length(dFdx(rel)), length(dFdy(rel))) / float(uDepthScale);
    if (!near && farDepth >= 1.0) {
        fragColor = vec4(1.0, -1.0, 0.0, 1.0);  // the sky
        return;
    }
    float dist = length(rel);
    float shade = 1.0;
    if ((uCloudShadowLayer >= 0 || uCloudShadowOther >= 0) && uCloudShadowSteps > 0
            && uOwnAtmosphere >= 0) {
        // The ground a VIEW pixel spans (the terrain's footprint, from its
        // position's derivatives): across the ray, its length over how
        // grazing the ray meets the ground (a level ground's), or, smaller on
        // a slope facing the camera, this pass's own derivatives over its
        // scale (huge across a silhouette: the level ground's there).
        vec3 planet = camPos + rel;
        vec3 dir = rel / max(dist, 1e-3);
        vec3 middle = unproject(toRender, vec3(ndc, 0.0));
        vec3 beside = unproject(toRender, vec3(ndc + vec2(2.0 / uViewSize.x, 0.0), 0.0));
        float angle = length(beside - middle) / max(length(middle - uCamRender), 1e-6)
                    / float(uDepthScale);
        float grazing = abs(dot(dir, normalize(planet)));
        float level = dist * angle / max(grazing, 0.02);
        shade = planetCloudShadow(planet, normalize(sunDir), uCloudShadowSteps,
                                  min(level, local));
    }
    fragColor = vec4(shade, dist / KM, 0.0, 1.0);
}
